# Required libraries:
# pip install psutil
# pip install nvidia-ml-py
# pip install pywin32

import psutil
import logging
import time
import datetime

try:
    import pynvml
    nvml = pynvml # Assign to nvml for consistency if successful
except ImportError:
    nvml = None  # Handle missing nvml gracefully

try:
    import win32evtlog
    import win32evtlogutil
    import pywintypes # Often needed for pywin32 error handling
    win_event_log_modules_available = True
except ImportError:
    win32evtlog = None
    win32evtlogutil = None
    pywintypes = None
    win_event_log_modules_available = False
    # Logging of this will be handled in functions using these modules

LOG_FILENAME = "system_monitor.log"
POLLING_INTERVAL = 5 # Polling interval in seconds. Adjust as needed.

# Set up logging
# Note: Log file will grow indefinitely. For long-term use, consider implementing log rotation.
# This can be done using Python's `logging.handlers.RotatingFileHandler` or `TimedRotatingFileHandler`.
logging.basicConfig(
    filename=LOG_FILENAME,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# NVML initialization and GPU handle retrieval
def initialize_nvml():
    logging.info("Attempting to initialize NVML...")
    if nvml:
        try:
            nvml.nvmlInit()
            handle = nvml.nvmlDeviceGetHandleByIndex(0) # Get handle for the first GPU
            logging.info("NVML initialized successfully and GPU handle retrieved.")
            return True, handle
        except nvml.NVMLError as e:
            logging.error(f"Error initializing NVML or getting GPU handle: {e}")
            # Attempt to shutdown NVML if init was partially successful before error
            try:
                nvml.nvmlShutdown()
            except nvml.NVMLError:
                pass # Ignore shutdown error if init failed badly
            return False, None
    else:
        logging.warning("NVML (pynvml) library not found. GPU monitoring will be disabled.")
        return False, None

def get_gpu_info(nvml_initialized, handle):
    logging.debug("Attempting to get GPU info.")
    if not nvml_initialized or not handle:
        return {'temperature': 'N/A', 'utilization': 'N/A', 'fan_speed': 'N/A', 'error': 'NVML not initialized or handle invalid'}

    gpu_info = {}
    try:
        temp = nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)
        gpu_info['temperature'] = f"{temp} C"
    except nvml.NVMLError as e:
        logging.error(f"Error getting GPU temperature: {e}")
        gpu_info['temperature'] = 'Error'

    try:
        util_rates = nvml.nvmlDeviceGetUtilizationRates(handle)
        gpu_info['utilization'] = f"{util_rates.gpu}%"
    except nvml.NVMLError as e:
        logging.error(f"Error getting GPU utilization: {e}")
        gpu_info['utilization'] = 'Error'

    try:
        fan_speed = nvml.nvmlDeviceGetFanSpeed(handle)
        gpu_info['fan_speed'] = f"{fan_speed}%"
    except nvml.NVMLError as e:
        # Some GPUs don't have a programmable fan or don't report speed
        if e.args[0] == nvml.NVML_ERROR_NOT_SUPPORTED:
            logging.warning(f"GPU fan speed not supported: {e}")
            gpu_info['fan_speed'] = 'N/A (Not Supported)'
        else:
            logging.error(f"Error getting GPU fan speed: {e}")
            gpu_info['fan_speed'] = 'Error'

    return gpu_info

def get_cpu_info():
    logging.debug("Attempting to get CPU info.")
    cpu_info = {}
    try:
        cpu_info['overall_usage'] = f"{psutil.cpu_percent(interval=1)}%"
        cpu_info['per_core_usage'] = [f"{core_usage}%" for core_usage in psutil.cpu_percent(interval=1, percpu=True)]
    except Exception as e:
        logging.error(f"Error getting CPU usage: {e}")
        cpu_info['overall_usage'] = 'Error'
        cpu_info['per_core_usage'] = 'Error'

    try:
        # CPU temperature readings depend on OS, hardware drivers (e.g., lm-sensors on Linux), and permissions.
        temps = psutil.sensors_temperatures()
        if temps:
            # Attempt to find core temperatures, structure varies by OS/hardware
            # This is a common pattern for Linux with k10temp or coretemp
            cpu_temps = {}
            for name, entries in temps.items():
                for entry in entries:
                    cpu_temps[entry.label or name] = f"{entry.current} C"
            cpu_info['temperatures'] = cpu_temps if cpu_temps else 'N/A'
        else:
            cpu_info['temperatures'] = 'N/A (No sensors found or not supported)'
    except Exception as e:
        logging.warning(f"Error getting CPU temperatures: {e}. Temperatures might not be available or supported.")
        cpu_info['temperatures'] = 'Error (accessing sensors failed)'

    return cpu_info

def get_ram_info():
    logging.debug("Attempting to get RAM info.")
    ram_info = {}
    try:
        mem = psutil.virtual_memory()
        ram_info['total'] = f"{mem.total / (1024**3):.2f} GB"
        ram_info['available'] = f"{mem.available / (1024**3):.2f} GB"
        ram_info['used'] = f"{mem.used / (1024**3):.2f} GB"
        ram_info['percent'] = f"{mem.percent}%"
    except Exception as e:
        logging.error(f"Error getting RAM info: {e}")
        ram_info = {key: 'Error' for key in ['total', 'available', 'used', 'percent']}
    return ram_info

def query_windows_event_logs(last_event_time=None):
    if not win_event_log_modules_available:
        # This warning is now logged once in the main loop.
        return last_event_time

    logging.debug(f"Querying Windows Event Logs. Looking for events after: {last_event_time}")
    server = 'localhost'
    log_types = ["System", "Application"]
    # Event type constants from win32con: EVENTLOG_ERROR_TYPE = 1, EVENTLOG_WARNING_TYPE = 2
    # We are looking for Error (1) events. "Critical" is often a sub-type or level of Error.

    # If last_event_time is None (first run), initialize it to fetch events from a defined period (e.g., last hour).
    # This is now handled by the initial value of last_event_log_check_time in __main__
    # but we keep a fallback here if the function is ever called with last_event_time=None directly.
    current_run_latest_event_time = last_event_time
    if current_run_latest_event_time is None:
        current_run_latest_event_time = datetime.datetime.now() - datetime.timedelta(hours=1)
        logging.info(f"No last_event_time provided, querying events from the last hour: {current_run_latest_event_time}")


    events_found_this_run_timestamps = []

    for log_type in log_types:
        try:
            handle = win32evtlog.OpenEventLog(server, log_type)
            # EVENTLOG_BACKWARDS_READ is important to get latest events first
            # EVENTLOG_SEQUENTIAL_READ ensures we read them in order from that point
            flags = win32evtlog.EVENTLOG_BACKWARDS_READ | win32evtlog.EVENTLOG_SEQUENTIAL_READ

            # Get total number of records to avoid error if log is empty, though ReadEventLog should handle it
            num_records = win32evtlog.GetNumberOfEventLogRecords(handle)
            if num_records == 0:
                win32evtlog.CloseEventLog(handle)
                continue

            raw_events = win32evtlog.ReadEventLog(handle, flags, 0)

            for raw_event in raw_events:
                # raw_event.TimeGenerated is a pywintypes.datetime object
                # Convert to standard Python datetime for comparison and storage
                event_time_generated = datetime.datetime.fromtimestamp(raw_event.TimeGenerated.timestamp())

                if event_time_generated <= current_run_latest_event_time:
                    # Since events are read backwards, once we see an event older than or same as last_event_time,
                    # all subsequent events in this log_type will also be older.
                    break

                if raw_event.EventType == 1: # EVENTLOG_ERROR_TYPE
                    source_name = raw_event.SourceName
                    event_id = raw_event.EventID & 0xFFFF # Mask to get lower 16 bits
                    message = win32evtlogutil.SafeFormatMessage(raw_event, log_type) # Use SafeFormatMessage

                    event_detail = (
                        f"Windows Event: LogType='{log_type}', Source='{source_name}', ID={event_id}, "
                        f"Type={raw_event.EventType}, Time='{event_time_generated.strftime('%Y-%m-%d %H:%M:%S')}', "
                        f"Message='{message.strip()}'"
                    )
                    logging.error(event_detail) # Log as ERROR level in our log file
                    events_found_this_run_timestamps.append(event_time_generated)

            win32evtlog.CloseEventLog(handle)

        except pywintypes.error as e:
            # Common errors: 5 (Access Denied), 2 (File Not Found - if log source is bad)
            logging.error(f"Windows Event Log API Error for '{log_type}': Code {e.args[0]} - {e.args[2]}")
            if not hasattr(query_windows_event_logs, 'access_warning_logged_api_error'): # Log permission warning once
                 logging.warning("Ensure the script has permissions to read Windows Event Logs if access denied errors persist.")
                 query_windows_event_logs.access_warning_logged_api_error = True
        except Exception as e:
            logging.error(f"Unexpected error querying Windows Event Log '{log_type}': {e}", exc_info=True)

    if events_found_this_run_timestamps:
        return max(events_found_this_run_timestamps) # Return the newest event time from this run
    return current_run_latest_event_time # Return the original time if no newer events were found

if __name__ == "__main__":
    # Explicit Startup Logging
    logging.info("System monitor starting up...")
    logging.info(f"Logging to: {LOG_FILENAME}") # Use the variable
    logging.info("Press Ctrl+C to stop monitoring.")

    nvml_initialized, gpu_handle = initialize_nvml()

    # Initialize last_event_log_check_time for the first run
    last_event_log_check_time = datetime.datetime.now() - datetime.timedelta(hours=1)

    # Ensure clean variable initialization before the loop
    cpu_data = {'overall_usage': 'N/A', 'temperatures': 'N/A'}
    ram_data = {'percent': 'N/A', 'used': 'N/A', 'total': 'N/A'}
    gpu_data = {'temperature': 'N/A', 'utilization': 'N/A', 'fan_speed': 'N/A'}


    try:
        while True:
            # It's good practice to call get_ functions and assign to fresh dicts
            # or ensure they return consistent structures even on error.
            # Current implementation of get_... functions already returns dicts with 'Error' strings.
            current_cpu_data = get_cpu_info()
            current_ram_data = get_ram_info()

            # Update main data dicts, preferring new data if available
            cpu_data.update(current_cpu_data)
            ram_data.update(current_ram_data)

            current_gpu_data = {}
            if nvml_initialized and gpu_handle:
                current_gpu_data = get_gpu_info(nvml_initialized, gpu_handle)
            gpu_data.update(current_gpu_data)


            # Refined Metric Logging
            # Use the potentially updated cpu_data, ram_data, gpu_data for logging
            cpu_temp_log_str = cpu_data.get('temperatures', 'N/A')
            if isinstance(cpu_temp_log_str, dict):
                core_temps = [v for k, v in cpu_temp_log_str.items() if 'core' in k.lower() or 'package' in k.lower()]
                if core_temps:
                    cpu_temp_log_str = ", ".join(core_temps)
                elif cpu_temp_log_str:
                     cpu_temp_log_str = next(iter(cpu_temp_log_str.values()))
                else:
                    cpu_temp_log_str = 'N/A'

            # Prepare GPU strings, indicating disabled status if NVML not initialized
            gpu_temp_log_str = gpu_data.get('temperature', 'N/A')
            gpu_util_log_str = gpu_data.get('utilization', 'N/A')
            gpu_fan_log_str = gpu_data.get('fan_speed', 'N/A')

            if not nvml_initialized:
                gpu_temp_log_str = 'N/A (Disabled)'
                gpu_util_log_str = 'N/A (Disabled)'
                gpu_fan_log_str = 'N/A (Disabled)'

            log_msg = (
                f"CPU: {cpu_data.get('overall_usage', 'N/A')} Usage, Temp: {cpu_temp_log_str} | "
                f"RAM: {ram_data.get('percent', 'N/A')} Used ({ram_data.get('used', 'N/A')} / {ram_data.get('total', 'N/A')}) | "
                f"GPU: Temp: {gpu_temp_log_str}, Util: {gpu_util_log_str}, Fan: {gpu_fan_log_str}"
            )
            logging.info(log_msg)

            # Windows Event Log Polling
            # Windows Event Log is polled every POLLING_INTERVAL. If this is too frequent or resource-intensive,
            # consider moving this call into a separate timer (e.g., poll every 30 or 60 seconds).
            # For example, by using a counter or checking `time.time()` against a `last_event_poll_time` variable.
            if win_event_log_modules_available:
                new_event_time = query_windows_event_logs(last_event_log_check_time)
                # new_event_time will be the timestamp of the latest event found in this run,
                # or the previous last_event_log_check_time if no new events were found.
                last_event_log_check_time = new_event_time
            elif not hasattr(query_windows_event_logs, 'warning_logged_module_missing'):
                logging.warning("Windows Event Log polling disabled: pywin32 library not found or failed to import.")
                query_windows_event_logs.warning_logged_module_missing = True # Log this warning only once

            # Polling interval defined by POLLING_INTERVAL constant at the top of the script
            time.sleep(POLLING_INTERVAL)

    except KeyboardInterrupt:
        logging.info("Monitoring stopped by user (KeyboardInterrupt).")
    except Exception as e:
        logging.error(f"An unexpected error occurred in the main loop: {e}", exc_info=True)
    finally:
        if nvml_initialized:
            try:
                nvml.nvmlShutdown()
                logging.info("NVML shut down successfully.")
            except nvml.NVMLError as e:
                logging.error(f"Error shutting down NVML: {e}")
