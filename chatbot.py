
import get_close_matches
from prettytable import PrettyTable
from datetime import datetime
from groq import Groq
import threading
import socket
import time
import os
import paho.mqtt.client as mqtt
from opcua import Client
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Add after your imports
class MaintenanceDataTracker:
    def __init__(self, window_size=4):
        self.window_size = window_size
        self.history = {
            'OEE': [],
            'performance': [],
            'cycles': [],
            'alarms': [],
            'stops': []
        }
        self.ideal_production_rate = None
        
    def add_reading(self, data, timestamp):
        try:
            oee = float(data.get('OEE', 0))
            performance = float(data.get('performance', 0))
            cycles = float(data.get('number of cycles', 0))
            alarms = float(data.get('alarm duration', 0))
            stops = float(data.get('stop duration', 0))
            
            self.history['OEE'].append((timestamp, oee))
            self.history['performance'].append((timestamp, performance))
            self.history['cycles'].append((timestamp, cycles))
            self.history['alarms'].append((timestamp, alarms))
            self.history['stops'].append((timestamp, stops))
            
            if self.ideal_production_rate is None or performance > self.ideal_production_rate:
                self.ideal_production_rate = performance
            
            self._cleanup_old_data(timestamp)
            
        except Exception as e:
            print(f"Error adding reading: {str(e)}")
    
    def _cleanup_old_data(self, current_time):
        cutoff = current_time - pd.Timedelta(hours=self.window_size)
        for metric in self.history:
            self.history[metric] = [(t, v) for t, v in self.history[metric] if t > cutoff]
    
    def get_stats(self):
        try:
            oee_values = [v for _, v in self.history['OEE']]
            oee_max = max(oee_values) if oee_values else 0
            oee_min = min(oee_values) if oee_values else 0
            oee_volatility = oee_max - oee_min
            
            latest_alarms = self.history['alarms'][-1][1] if self.history['alarms'] else 0
            latest_stops = self.history['stops'][-1][1] if self.history['stops'] else 0
            latest_cycles = self.history['cycles'][-1][1] if self.history['cycles'] else 0
            
            return {
                'OEE_4hour_max': oee_max,
                'OEE_4hour_min': oee_min,
                'OEE_volatility': oee_volatility,
                'A_Alarm_Duration': latest_alarms,
                'A_Stop_Duration': latest_stops,
                'P_Ideal_Production_Rate': self.ideal_production_rate or 0,
                'P_No_of_Cycles': latest_cycles
            }
            
        except Exception as e:
            print(f"Error calculating stats: {str(e)}")
            return None



# Set up environment variable for Groq API key
os.environ["GROQ_API_KEY"] = "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"

node_ids = {
    "temperature": "ns=4;s=|var|Turck/ARM/WinCE TV.Application.GVL_AI_Scaled.MT_AI2_Temp_Scaled",
    "flow rate": "ns=4;s=|var|Turck/ARM/WinCE TV.Application.GVL_AI_Scaled.Comp_Air_Flow_Rate_LPM_SCALED",
    "pH": "ns=4;s=|var|Turck/ARM/WinCE TV.Application.GVL_AI_Scaled.MT_AI1_PH_Scaled",
    "OEE": {
        "OEE": "ns=4;s=|var|Turck/ARM/WinCE TV.Application.GVL_OEE.OEE",
        "quality": "ns=4;s=|var|Turck/ARM/WinCE TV.Application.GVL_OEE.OEE_Quality",
        "performance": "ns=4;s=|var|Turck/ARM/WinCE TV.Application.GVL_OEE.OEE_Performance",
        "availability": "ns=4;s=|var|Turck/ARM/WinCE TV.Application.GVL_OEE.OEE_Availability",
        "alarm duration": "ns=4;s=|var|Turck/ARM/WinCE TV.Application.GVL_OEE.A_Alarm_Duration",
        "idle duration": "ns=4;s=|var|Turck/ARM/WinCE TV.Application.GVL_OEE.A_Idle_Duration",
        "manual duration": "ns=4;s=|var|Turck/ARM/WinCE TV.Application.GVL_OEE.A_Manual_Duration",
        "run duration": "ns=4;s=|var|Turck/ARM/WinCE TV.Application.GVL_OEE.A_Run_Duration",
        "stop duration": "ns=4;s=|var|Turck/ARM/WinCE TV.Application.GVL_OEE.A_Stop_Duration",
        "total duration": "ns=4;s=|var|Turck/ARM/WinCE TV.Application.GVL_OEE.A_Total_Duration",
        "average cycle time": "ns=4;s=|var|Turck/ARM/WinCE TV.Application.GVL_OEE.P_Average_Time",
        "current cycle time": "ns=4;s=|var|Turck/ARM/WinCE TV.Application.GVL_OEE.P_Current_Time",
        "number of cycles": "ns=4;s=|var|Turck/ARM/WinCE TV.Application.GVL_OEE.P_Current_Time",
        "previous cycle time": "ns=4;s=|var|Turck/ARM/WinCE TV.Application.GVL_OEE.P_No_Of_Cycles",
        "bad bottles": "ns=4;s=|var|Turck/ARM/WinCE TV.Application.GVL_OEE.Q_No_Of_Bad",
        "good bottles": "ns=4;s=|var|Turck/ARM/WinCE TV.Application.GVL_OEE.Q_No_Of_Good",
        "total bottles": "ns=4;s=|var|Turck/ARM/WinCE TV.Application.GVL_OEE.Q_Total_No"
            }
    }


# Define quit keywords
quit_keywords = {"exit", "quit", "bye", "goodbye"}

# Check internet connection
def check_internet():
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=2)  # Increased timeout to 5 seconds
        return True
    except OSError:
        return False


def spell_correct(user_input, valid_keywords):
    exclusions = {"pH", "OEE"}
    # Assuming you use a spell checker like pyspellchecker or fuzzy matching
    corrected_input = user_input.lower().strip()

    # Example: Using fuzzy matching or direct keyword check
    closest_match = None
    for keyword in valid_keywords:
        if corrected_input in keyword.lower():
            closest_match = keyword
            break

    # Return the closest match or the original input if no match is found
    return closest_match if closest_match else corrected_input


# Getting nodes
def fetch_live_data(query, max_retries=3):
    """Fetch and format current machine data with retry logic"""
    server_url = "opc.tcp://192.168.250.11:4840/"
    client = None
    
    for attempt in range(max_retries):
        try:
            client = Client(server_url)
            client.connection_timeout = 10000  # Increase timeout to 10 seconds
            client.connect()
            results = {}
            
            # If querying "OEE", fetch all OEE components
            if "oee" in query.lower():
                for key, node_id in node_ids["OEE"].items():
                    try:
                        node = client.get_node(node_id)
                        value = node.get_value()
                        results[key] = f"{value}"
                    except Exception as e:
                        print(f"Warning: Failed to fetch {key}: {str(e)}")
                        results[key] = "0"  # Default value on error
                return results
            
            # Fetch individual data points
            for key, node_id in node_ids.items():
                if key in query.lower() and not isinstance(node_id, dict):
                    node = client.get_node(node_id)
                    value = node.get_value()
                    return {key: f"{value:.2f}"}
            
            return "I'm sorry, I couldn't find the requested data."
            
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                print("Retrying in 2 seconds...")
                time.sleep(2)  # Wait 2 seconds before retry
            else:
                return {"error": f"Failed after {max_retries} attempts: {str(e)}"}
        
        finally:
            if client:
                try:
                    client.disconnect()
                except Exception as e:
                    pass  # Ignore disconnect errors



# Pretty table display
def display_oee_table(oee_data):
    # Create tables
    availability_table = PrettyTable()
    performance_table = PrettyTable()
    quality_table = PrettyTable()

    # Populate "Availability" table
    availability_table.title = "OEE: Availability"
    availability_table.field_names = ["Component", "Value"]
    availability_components = ["availability", "stop duration", "idle duration", "alarm duration", "run duration", "manual duration"]
    for component in availability_components:
        availability_table.add_row([component.capitalize(), oee_data.get(component, "N/A")])

    # Populate "Performance" table
    performance_table.title = "OEE: Performance"
    performance_table.field_names = ["Component", "Value"]
    performance_components = [
        "current cycle time",
        "previous cycle time", 
        "number of cycles",
        "average cycle time"
    ]  # Removed "min cycle time" since it's not in the data
    for component in performance_components:
        performance_table.add_row([component.capitalize(), oee_data.get(component, "N/A")])

    # Populate "Quality" table
    quality_table.title = "OEE: Quality"
    quality_table.field_names = ["Component", "Value"]
    quality_components = ["good products", "bad products", "total number"]
    for component in quality_components:
        quality_table.add_row([component.capitalize(), oee_data.get(component, "N/A")])

    # Print tables
    print("\n")
    print(availability_table)
    print("\n")
    print(performance_table)
    print("\n")
    print(quality_table)




# Format and display OEE data
def format_live_data_response(data):
    response = "Here is the data you requested:\n"
    for key, value in data.items():
        response += f"{key.capitalize()}: {value}\n"
    return response


def data_query(user_input):
    # This function checks if the query involves live data and handles it accordingly
    live_data_keywords = ['temperature', 'flow rate', 'ph', 'oee']
    for keyword in live_data_keywords:
        if keyword in user_input.lower():
            return fetch_live_data(user_input)
    return None

def ai_response(prompt):
    if check_internet():
        try:
            client = Groq()
            completion = client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=512,
                top_p=1,
                stream=True
            )
            response_text = ""
            for chunk in completion:
                response_text += chunk.choices[0].delta.content or ""
            return response_text
        except Exception as e:
            return f"An error occurred: {str(e)}"




# Global variables with proper initialization
maintenance_tracker = None
model = None

def initialize_chatbot():
    """Initialize global variables and models"""
    global maintenance_tracker, model
    try:
        maintenance_tracker = MaintenanceDataTracker()
        try:
            model = joblib.load('enhanced_oee_maintenance_predictor.joblib')
        except Exception as e:
            print(f"Warning: Could not load model: {str(e)}")
            model = None
    except Exception as e:
        print(f"Error initializing chatbot: {str(e)}")
        raise

# Initialize when module is imported
try:
    initialize_chatbot()
except Exception as e:
    print(f"Failed to initialize chatbot: {str(e)}")

def get_current_machine_data(for_prediction=False):
    """
    Fetch and format current machine data
    for_prediction: if True, return only features needed for model prediction
    """
    try:
        data = fetch_live_data("oee")
        
        if isinstance(data, dict) and "error" not in data:
            def safe_float(value, default=0.0):
                try:
                    return float(value)
                except (ValueError, TypeError):
                    return default

            def safe_divide(a, b, default=0.0):
                try:
                    return float(a) / float(b) if float(b) != 0 else default
                except (ValueError, TypeError):
                    return default

            # Calculate all metrics
            total_duration = safe_float(data.get('total duration', 0))
            alarm_duration = safe_float(data.get('alarm duration', 0))
            stop_duration = safe_float(data.get('stop duration', 0))
            run_duration = safe_float(data.get('run duration', 0))
            idle_duration = safe_float(data.get('idle duration', 0))
            
            good_bottles = safe_float(data.get('good bottles', 0))
            bad_bottles = safe_float(data.get('bad bottles', 0))
            total_bottles = safe_float(data.get('total bottles', 0))
            
            # Calculate rates and ratios
            alarm_rate = safe_divide(alarm_duration, total_duration)
            downtime_ratio = safe_divide(stop_duration, total_duration)
            stop_rate = safe_divide(stop_duration, run_duration)
            reject_rate = safe_divide(bad_bottles, total_bottles)
            
            oee_value = safe_float(data.get('OEE', 0))
            quality_rate = safe_float(data.get('quality', 0))
            cycle_efficiency = safe_float(data.get('performance', 0))
            
            ideal_cycle_time = safe_float(data.get('average cycle time', 0))
            current_cycle_time = safe_float(data.get('current cycle time', 0))
            speed_loss = safe_divide(current_cycle_time - ideal_cycle_time, ideal_cycle_time) if ideal_cycle_time > 0 else 0
            
            reject_rate_change = 0.0

            # Create full dataset with all metrics
            all_metrics = {
                # Model required features
                'OEE_4hour_max': oee_value,
                'OEE_4hour_min': oee_value,
                'OEE_volatility': 0.0,
                'A_Alarm_Duration': alarm_duration,
                'A_Stop_Duration': stop_duration,
                'A_Run_Duration': run_duration,
                'Alarm_Rate': alarm_rate,
                'Downtime_Ratio': downtime_ratio,
                'Cycle_Efficiency': cycle_efficiency,
                'Quality_Rate': quality_rate,
                'P_Ideal_Production_Rate': cycle_efficiency,
                'P_No_of_Cycles': safe_float(data.get('number of cycles', 0)),
                'Q_No_of_Bad': bad_bottles,
                'Q_No_of_Good': good_bottles,
                'Reject_Rate_Change': reject_rate_change,
                'Speed_Loss': speed_loss,
                'Stop_Rate': stop_rate,
                
                # Additional metrics for display
                'Q_Total_No': total_bottles,
                'Idle_Duration': idle_duration,
                'Current_Cycle_Time': current_cycle_time,
                'Ideal_Cycle_Time': ideal_cycle_time,
                'Total_Duration': total_duration
            }

            # If for prediction, only return model features
            if for_prediction:
                model_features = [
                    'A_Alarm_Duration', 'A_Run_Duration', 'A_Stop_Duration', 
                    'Alarm_Rate', 'Cycle_Efficiency', 'Downtime_Ratio', 
                    'OEE_4hour_max', 'OEE_4hour_min', 'OEE_volatility',
                    'P_Ideal_Production_Rate', 'P_No_of_Cycles', 
                    'Q_No_of_Bad', 'Q_No_of_Good', 'Quality_Rate',
                    'Reject_Rate_Change', 'Speed_Loss', 'Stop_Rate'
                ]
                metrics_for_model = {k: v for k, v in all_metrics.items() if k in model_features}
                return pd.DataFrame(metrics_for_model, index=[0])
            
            # Return all metrics for display
            return pd.DataFrame(all_metrics, index=[0])
            
        else:
            print("Error: Invalid or missing OEE data")
            print("Raw data received:", data)
            return None
            
    except Exception as e:
        print(f"Error in get_current_machine_data: {str(e)}")
        import traceback
        print("Traceback:", traceback.format_exc())
        return None


def predict_maintenance(current_data):
    """Make maintenance predictions using the loaded model"""
    try:
        # Make prediction (scaling is handled within the model pipeline)
        prediction = model.predict(current_data)[0]  # Get first prediction
        probability = model.predict_proba(current_data)[0]  # Get probability scores
        
        return prediction, probability
    except Exception as e:
        print(f"Error making prediction: {str(e)}")
        return None, None


def handle_maintenance_query(user_input):
    """Handle maintenance predictions using the ML model"""
    try:
        # Get data for prediction
        current_data = get_current_machine_data(for_prediction=True)
        
        # Get full data for display
        display_data = get_current_machine_data(for_prediction=False)
        
        if current_data is not None:
            prediction, probability = predict_maintenance(current_data)
            
            # Display full metrics if requested
            if "show data" in user_input.lower() or "details" in user_input.lower():
                print("\nCurrent Machine Metrics:")
                print(display_data)
            
            return format_prediction_response(prediction, probability)
        return "Unable to fetch current machine data"
        
    except Exception as e:
        print(f"Error in maintenance prediction: {str(e)}")
        return "Error making maintenance prediction"

def display_maintenance_prediction(prediction_response):
    """Format and display maintenance prediction"""
    table = PrettyTable()
    table.title = "Maintenance Prediction"
    table.field_names = ["Metric", "Value"]
    
    # Parse the response string into lines
    lines = prediction_response.split('\n')
    
    for line in lines:
        if line.strip():  # Skip empty lines
            if ':' in line:
                key, value = line.split(':', 1)
                table.add_row([key.strip(), value.strip()])
            elif '•' in line:
                table.add_row(["Recommendation", line.strip('• ')])
            else:
                table.add_row(['Note', line.strip()])
    
    print("\n")
    print(table)

def handle_oee_query(user_input, oee_data):
    """Handle specific OEE component queries"""
    user_input = user_input.lower()
    
    # Check for specific component queries
    for component in oee_data.keys():
        if component.lower() in user_input:
            return f"{component.capitalize()}: {oee_data[component]}"
    
    # If no specific component mentioned, display full OEE table
    display_oee_table(oee_data)
    return None

def get_risk_level(probability):
    """Convert probability to risk level"""
    try:
        prob_value = float(probability)
        if prob_value >= 0.7:
            return "HIGH", "🔴"
        elif prob_value >= 0.3:
            return "MEDIUM", "🟡"
        else:
            return "LOW", "🟢"
    except (ValueError, TypeError):
        print(f"Warning: Invalid probability value: {probability}")
        return "UNKNOWN", "⚪"



# Update chatbot function to include detailed maintenance analysis
def chatbot(user_input):
    """
    Modified version of chatbot to handle single message interactions
    Returns: string response for the user
    """
    try:
        global maintenance_tracker
        if maintenance_tracker is None:
            maintenance_tracker = MaintenanceDataTracker()
        
        # Handle quit commands
        corrected_query = spell_correct(user_input, quit_keywords)
        if corrected_query.lower() in quit_keywords:
            return "Goodbye!"
        
        # Handle maintenance queries
        if "maintenance" in user_input.lower():
            current_data = get_current_machine_data()
            if current_data is not None:
                maintenance_response = handle_maintenance_query(current_data)
                return maintenance_response
            return "Sorry, I couldn't fetch the current machine data."
        
        # Handle live data queries
        corrected_query1 = spell_correct(user_input, node_ids.keys())
        if any(key in corrected_query1.lower() for key in node_ids):
            live_data = fetch_live_data(corrected_query1)
            
            if isinstance(live_data, dict):
                if "error" in live_data:
                    return f"Unable to retrieve data. {live_data['error']}"
                elif "oee" in corrected_query1.lower():
                    specific_response = handle_oee_query(user_input, live_data)
                    return specific_response if specific_response else format_live_data_response(live_data)
                else:
                    return format_live_data_response(live_data)
            return str(live_data)
        
        # Default to AI response
        return ai_response(user_input)
        
    except Exception as e:
        return f"An error occurred: {str(e)}"