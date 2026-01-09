import os
import time
import math
from datetime import datetime, timedelta, timezone
from collections import Counter
from flask import Flask, redirect, url_for, session, request, render_template, jsonify
from stravalib.client import Client
from dotenv import load_dotenv
import folium
from folium.plugins import HeatMap
import polyline
from timezonefinder import TimezoneFinder
from zoneinfo import ZoneInfo

load_dotenv()

app = Flask(__name__)

# Initialize timezone finder (singleton for performance)
_tf = TimezoneFinder()


def get_timezone_for_location(lat, lng):
    """Get timezone string for given coordinates."""
    if lat is None or lng is None:
        return None
    tz_str = _tf.timezone_at(lat=lat, lng=lng)
    return tz_str


def convert_to_local_time(dt, tz_str):
    """Convert a datetime to the specified timezone.
    
    Args:
        dt: datetime object (can be naive or aware)
        tz_str: timezone string like 'Europe/Budapest'
    
    Returns:
        datetime in the specified timezone, or original if conversion fails
    """
    if dt is None or tz_str is None:
        return dt
    try:
        tz = ZoneInfo(tz_str)
        # If datetime is naive, assume it's UTC
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(tz)
    except Exception:
        return dt
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'default_secret_key')

CLIENT_ID = os.getenv('STRAVA_CLIENT_ID')
CLIENT_SECRET = os.getenv('STRAVA_CLIENT_SECRET')
REDIRECT_URI = 'http://localhost:5000/authorization'

client = Client()

def get_strava_client():
    if 'access_token' not in session:
        return None
    
    # Check if token is expired
    if time.time() > session.get('expires_at', 0):
        refresh_response = client.refresh_access_token(
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET,
            refresh_token=session['refresh_token']
        )
        session['access_token'] = refresh_response['access_token']
        session['refresh_token'] = refresh_response['refresh_token']
        session['expires_at'] = refresh_response['expires_at']
        
    return Client(access_token=session['access_token'])

@app.route('/')
def index():
    if 'access_token' not in session:
        authorize_url = client.authorization_url(
            client_id=CLIENT_ID,
            redirect_uri=REDIRECT_URI,
            scope=['read', 'activity:read_all', 'activity:write']
        )
        return render_template('index.html', authorize_url=authorize_url)
    return redirect(url_for('monthly_stats'))

@app.route('/authorization')
def authorization():
    code = request.args.get('code')
    token_response = client.exchange_code_for_token(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        code=code
    )
    session['access_token'] = token_response['access_token']
    session['refresh_token'] = token_response['refresh_token']
    session['expires_at'] = token_response['expires_at']
    return redirect(url_for('monthly_stats'))

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

def identify_locations(activities):
    """Identify home and work locations using simple grid-based clustering.
    
    Points are rounded to ~100m precision and the two most frequent locations
    are identified. Then, departure times are analyzed to determine which is
    home (earlier departures = "to work") and which is work (later departures = "to home").
    """
    points = []
    for act in activities:
        if act.start_latlng:
            lat = float(act.start_latlng.root[0])
            lng = float(act.start_latlng.root[1])
            points.append((lat, lng))
        if act.end_latlng:
            lat = float(act.end_latlng.root[0])
            lng = float(act.end_latlng.root[1])
            points.append((lat, lng))
    
    if not points:
        return None, None
    
    # Round coordinates to ~100m precision (3 decimal places ≈ 111m)
    rounded_points = [(round(lat, 3), round(lng, 3)) for lat, lng in points]
    
    # Count occurrences of each rounded location
    counter = Counter(rounded_points)
    most_common = counter.most_common(2)
    
    if len(most_common) < 2:
        # Not enough distinct locations found
        return None, None
    
    # Initially assign the two most frequent locations as loc_a and loc_b
    loc_a = list(most_common[0][0])
    loc_b = list(most_common[1][0])
    
    # Use departure times to determine which is home vs work
    # Home: where you leave FROM in the morning (to go to work)
    # Work: where you leave FROM in the evening (to go home)
    home, work = _assign_home_work_by_time(activities, loc_a, loc_b)
    
    return home, work


def _assign_home_work_by_time(activities, loc_a, loc_b):
    """Determine which location is home and which is work based on departure times.
    
    People typically leave home in the morning (earlier) and leave work in the evening (later).
    We analyze the average departure times from each location to make this determination.
    """
    # Collect departure times (minutes since midnight) from each location
    times_from_a = []
    times_from_b = []
    
    for act in activities:
        if act.start_latlng is None or not is_workday_with_cutoff(act.start_date, cutoff_hour=3):
            continue
        
        start_coords = act.start_latlng.root if hasattr(act.start_latlng, 'root') else act.start_latlng
        start_rounded = (round(float(start_coords[0]), 3), round(float(start_coords[1]), 3))
        
        minutes_since_midnight = act.start_date.hour * 60 + act.start_date.minute
        
        # Check which location this activity starts from
        if start_rounded == tuple(loc_a):
            times_from_a.append(minutes_since_midnight)
        elif start_rounded == tuple(loc_b):
            times_from_b.append(minutes_since_midnight)
    
    # Calculate average departure times
    avg_from_a = sum(times_from_a) / len(times_from_a) if times_from_a else None
    avg_from_b = sum(times_from_b) / len(times_from_b) if times_from_b else None
    
    # If we can't determine times, fall back to original order
    if avg_from_a is None or avg_from_b is None:
        return loc_a, loc_b
    
    # Home is where you leave earlier (morning commute to work)
    # Work is where you leave later (evening commute to home)
    if avg_from_a < avg_from_b:
        # loc_a has earlier departures -> it's home
        return loc_a, loc_b
    else:
        # loc_b has earlier departures -> it's home
        return loc_b, loc_a

def is_near(p1, p2, threshold_km=1.0):
    if p1 is None or p2 is None:
        return False
    
    # Handle stravalib.model.LatLon which might not be directly subscriptable
    p1_coords = p1.root if hasattr(p1, 'root') else p1
    p2_coords = p2.root if hasattr(p2, 'root') else p2

    # Simple Euclidean distance for small distances (approximate)
    dlat = (p1_coords[0] - p2_coords[0]) * 111
    dlng = (p1_coords[1] - p2_coords[1]) * 111 * math.cos(math.radians(p1_coords[0]))
    return math.sqrt(dlat**2 + dlng**2) < threshold_km


def is_workday_with_cutoff(dt: datetime, cutoff_hour: int = 3) -> bool:
    """Return True if `dt` should be treated as a workday (Mon-Fri).

    The day boundary is shifted by `cutoff_hour`, so e.g. 01:00 on Saturday is
    treated as Friday when `cutoff_hour` is 3.
    """
    if dt is None:
        return False
    effective_dt = dt - timedelta(hours=cutoff_hour)
    # Monday=0 .. Sunday=6
    return effective_dt.weekday() < 5

def analyze_commutes(activities, home, work):
    commutes = []
    commute_activity_ids = set()
    if home is None or work is None:
        return [], commute_activity_ids

    # Sort activities by start time
    activities.sort(key=lambda x: x.start_date)
    
    i = 0
    while i < len(activities):
        act = activities[i]

        # Weekend filter (with 03:00 cutoff): weekend rides are not commutes.
        # Example: Fri 01:00 (after midnight) still counts as Friday.
        if not is_workday_with_cutoff(act.start_date, cutoff_hour=3):
            i += 1
            continue
        
        # Check if direct commute
        is_h_to_w = is_near(act.start_latlng, home) and is_near(act.end_latlng, work)
        is_w_to_h = is_near(act.start_latlng, work) and is_near(act.end_latlng, home)
        
        if is_h_to_w or is_w_to_h:
            commutes.append({
                'activities': [act],
                'type': 'direct',
                'direction': 'to_work' if is_h_to_w else 'to_home',
                'distance': float(act.distance),
                'date': act.start_date
            })
            commute_activity_ids.add(act.id)
            i += 1
            continue
            
        # Check for chained activities
        chain = []
        current_start = act.start_latlng
        if is_near(current_start, home) or is_near(current_start, work):
            target = work if is_near(current_start, home) else home
            chain.append(act)
            temp_i = i + 1
            reached_target = is_near(act.end_latlng, target)
            
            while temp_i < len(activities) and not reached_target:
                next_act = activities[temp_i]

                # If the next activity is on a weekend (using the same cutoff),
                # don't allow it to contribute to a commute chain.
                if not is_workday_with_cutoff(next_act.start_date, cutoff_hour=3):
                    break

                # Check if next activity starts reasonably soon and continues the journey
                elapsed = chain[-1].elapsed_time
                if hasattr(elapsed, 'timedelta'):
                    elapsed = elapsed.timedelta()
                
                time_diff = next_act.start_date - chain[-1].start_date - elapsed
                if time_diff < timedelta(hours=8): # Allow for long stops/detours
                    chain.append(next_act)
                    if is_near(next_act.end_latlng, target):
                        reached_target = True
                else:
                    break
                temp_i += 1
            
            if reached_target:
                # Determine direction based on starting point
                chain_direction = 'to_work' if is_near(current_start, home) else 'to_home'
                commutes.append({
                    'activities': chain,
                    'type': 'chained',
                    'direction': chain_direction,
                    'distance': sum(float(a.distance) for a in chain),
                    'date': chain[0].start_date
                })
                for a in chain:
                    commute_activity_ids.add(a.id)
                # For distance estimation: "these chained activities should not be part of the commute distance estimation"
                i = temp_i
                continue
        
        i += 1
    return commutes, commute_activity_ids


def _self_test_weekend_cutoff():
    # Fri 01:00 should still be treated as Friday (workday) with 03:00 cutoff.
    assert is_workday_with_cutoff(datetime(2026, 1, 9, 1, 0), cutoff_hour=3) is True
    # Sat 04:00 should be treated as Saturday (weekend) with 03:00 cutoff.
    assert is_workday_with_cutoff(datetime(2026, 1, 10, 4, 0), cutoff_hour=3) is False

@app.route('/stats')
def monthly_stats():
    sc = get_strava_client()
    if not sc:
        return redirect(url_for('index'))
    
    selected_month = request.args.get('month')
    if not selected_month:
        selected_month = datetime.now().strftime('%Y-%m')
    
    # Identify home/work locations using a wider range for better results (e.g., last 180 days)
    after_locations = datetime.now() - timedelta(days=180)
    all_activities_for_id = list(sc.get_activities(after=after_locations, limit=200))
    all_activities_for_id = [a for a in all_activities_for_id if a.type in ['Ride', 'VirtualRide']]
    home, work = identify_locations(all_activities_for_id)
    
    # Get available months from the last 2 years
    available_months = []
    curr = datetime.now().replace(day=1)
    for _ in range(24):
        available_months.append(curr.strftime('%Y-%m'))
        curr = (curr - timedelta(days=1)).replace(day=1)

    # Fetch activities for the selected month
    after = datetime.strptime(selected_month, '%Y-%m')
    before = (after + timedelta(days=32)).replace(day=1)
    
    activities = list(sc.get_activities(after=after, before=before, limit=200))
    activities = [a for a in activities if a.type in ['Ride', 'VirtualRide']]
    
    commutes, commute_ids = analyze_commutes(activities, home, work)
    
    # Get timezone from home location for displaying local times
    home_tz_str = get_timezone_for_location(home[0], home[1]) if home else None
    
    stats = {}
    month_key = selected_month
    stats[month_key] = {
        'count': 0, 
        'distance': 0.0, 
        'commutes': [],
        'all_activities': [],
        'timezone': home_tz_str  # Store timezone for display in template
    }
    
    # Build the list of all activities for display, grouping chained commutes
    all_display_items = []
    processed_activity_ids = set()

    # First, add grouped commutes
    for c in commutes:
        ids = [a.id for a in c['activities']]
        names = [a.name for a in c['activities']]
        # Convert date to local time for display
        local_date = convert_to_local_time(c['date'], home_tz_str) if home_tz_str else c['date']
        all_display_items.append({
            'ids': ids,
            'names': names,
            'is_commute': True,
            'type': c['type'],
            'date': c['date'],  # Keep original for sorting
            'local_date': local_date,  # Local time for display
            'distance': c['distance']
        })
        for aid in ids:
            processed_activity_ids.add(aid)

    # Then, add remaining non-commute activities
    for act in activities:
        if act.id not in processed_activity_ids:
            local_date = convert_to_local_time(act.start_date, home_tz_str) if home_tz_str else act.start_date
            all_display_items.append({
                'ids': [act.id],
                'names': [act.name],
                'is_commute': False,
                'type': None,
                'date': act.start_date,  # Keep original for sorting
                'local_date': local_date,  # Local time for display
                'distance': float(act.distance)
            })

    # Sort by date
    all_display_items.sort(key=lambda x: x['date'])
    stats[month_key]['all_activities'] = all_display_items

    for c in commutes:
        stats[month_key]['count'] += 1
        if c['type'] == 'direct':
            stats[month_key]['distance'] += c['distance']
        stats[month_key]['commutes'].append(c)
        
    # Average commute distance
    direct_commutes = [c for c in stats[month_key]['commutes'] if c['type'] == 'direct']
    if direct_commutes:
        stats[month_key]['avg_dist'] = sum(c['distance'] for c in direct_commutes) / len(direct_commutes)
    else:
        stats[month_key]['avg_dist'] = 0
    
    # Calculate average departure times and standard deviations
    to_work_times = []  # minutes since midnight (in local time)
    to_home_times = []
    
    for c in stats[month_key]['commutes']:
        departure = c['date']
        # Convert to home timezone for accurate local time display
        if home_tz_str:
            departure = convert_to_local_time(departure, home_tz_str)
        minutes_since_midnight = departure.hour * 60 + departure.minute
        if c.get('direction') == 'to_work':
            to_work_times.append(minutes_since_midnight)
        elif c.get('direction') == 'to_home':
            to_home_times.append(minutes_since_midnight)
    
    def format_time(minutes):
        """Convert minutes since midnight to HH:MM format."""
        h = int(minutes) // 60
        m = int(minutes) % 60
        return f"{h:02d}:{m:02d}"
    
    def calc_time_stats(times_list):
        """Calculate average and std dev for a list of times (in minutes)."""
        if not times_list:
            return None, None
        avg = sum(times_list) / len(times_list)
        if len(times_list) > 1:
            variance = sum((x - avg) ** 2 for x in times_list) / len(times_list)
            std = math.sqrt(variance)
        else:
            std = 0
        return avg, std
    
    to_work_avg, to_work_std = calc_time_stats(to_work_times)
    to_home_avg, to_home_std = calc_time_stats(to_home_times)
    
    stats[month_key]['to_work_avg'] = format_time(to_work_avg) if to_work_avg is not None else None
    stats[month_key]['to_work_std'] = round(to_work_std) if to_work_std is not None else None  # in minutes
    stats[month_key]['to_work_count'] = len(to_work_times)
    stats[month_key]['to_home_avg'] = format_time(to_home_avg) if to_home_avg is not None else None
    stats[month_key]['to_home_std'] = round(to_home_std) if to_home_std is not None else None  # in minutes
    stats[month_key]['to_home_count'] = len(to_home_times)
            
    stats[month_key]['all_activities'].sort(key=lambda x: x['date'])

    return render_template('stats.html', stats=stats, home=home, work=work, 
                           available_months=available_months, selected_month=selected_month)

@app.route('/map/<month>')
def commute_map(month):
    sc = get_strava_client()
    if not sc: return redirect(url_for('index'))
    
    selected_ids = request.args.get('ids', '')
    if selected_ids:
        selected_ids = set(int(aid) for aid in selected_ids.split(','))
    else:
        selected_ids = None

    # Re-fetch activities to get map data (polylines)
    after_month = datetime.strptime(month, '%Y-%m')
    next_month = (after_month + timedelta(days=32)).replace(day=1)
    activities = list(sc.get_activities(after=after_month, before=next_month, limit=200))
    activities = [a for a in activities if a.type in ['Ride', 'VirtualRide']]
    
    # Identify locations based on a wider range for consistency
    after_locations = datetime.now() - timedelta(days=180)
    all_activities_for_id = list(sc.get_activities(after=after_locations, limit=200))
    all_activities_for_id = [a for a in all_activities_for_id if a.type in ['Ride', 'VirtualRide']]
    home, work = identify_locations(all_activities_for_id)

    # Filter for selected activities for heatmap
    if selected_ids is not None:
        activities = [a for a in activities if a.id in selected_ids]
    
    all_points = []
    m = None
    
    for act in activities:
        # get_activities doesn't return detailed map by default, we need to fetch detailed activity or assume summary_polyline exists
        if act.map and act.map.summary_polyline:
            points = polyline.decode(act.map.summary_polyline)
            all_points.extend(points)
            
            if m is None and points:
                m = folium.Map(location=points[0], zoom_start=13)
    
    if m is None:
        if home is not None:
            m = folium.Map(location=home, zoom_start=13)
        else:
            # Try to get center of all points if no map yet
            if all_points:
                m = folium.Map(location=all_points[0], zoom_start=13)
            else:
                return "No GPS data found for selected activities in this month."

    if all_points:
        # radius=10 and blur=10 helps to differentiate between frequent and infrequent routes.
        # Frequent routes will overlap and appear warmer, while single trips remain as 'colder' traces.
        HeatMap(all_points, radius=10, blur=10, min_opacity=0.3).add_to(m)
        
    if home is not None:
        folium.Marker(home, popup='Home', icon=folium.Icon(color='green')).add_to(m)
    if work is not None:
        folium.Marker(work, popup='Work', icon=folium.Icon(color='red')).add_to(m)

    return m._repr_html_()

@app.route('/mass_edit', methods=['POST'])
def mass_edit():
    sc = get_strava_client()
    if not sc: return jsonify({'error': 'Not authenticated'}), 401
    
    activity_ids = request.form.getlist('activity_ids')
    for aid in activity_ids:
        # Update to commute
        sc.update_activity(aid, commute=True)
        
        # Visibility "followers_only" and "private" check
        # stravalib might not expose 'visibility' directly in update_activity.
        # We might need to use a raw PUT request to https://www.strava.com/api/v3/activities/{id}
        # with 'visibility': 'followers_only'
        
        # First check current visibility/private status
        # activity = sc.get_activity(aid)
        # if activity.private:
        #     # stravalib update_activity doesn't seem to have visibility. 
        #     # Using requests to hit the API directly.
        
        access_token = session['access_token']
        import requests as r
        headers = {'Authorization': f'Bearer {access_token}'}
        # To set visibility to followers_only, we can try this:
        r.put(f'https://www.strava.com/api/v3/activities/{aid}', 
              headers=headers, 
              data={'visibility': 'followers_only'})
    
    month = request.form.get('month')
    return redirect(url_for('monthly_stats', month=month))

if __name__ == '__main__':
    if os.getenv('RUN_SELF_TESTS') == '1':
        _self_test_weekend_cutoff()
    app.run(debug=True)
