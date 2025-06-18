# System Event and Performance Monitor

This script monitors system performance metrics (CPU, RAM, GPU) and logs them along with critical Windows Event Log errors to a local file. It is designed to help diagnose system stability issues, such as unexpected crashes or performance degradation.

## Prerequisites

*   Python 3.x
*   Windows Operating System (for full functionality, including GPU monitoring via NVML and Windows Event Log polling)

## Setup

1.  **Install Python:** If you don't have Python installed, download it from [python.org](https://www.python.org/).
2.  **Download Script:** Save the `system_monitor.py` script to a directory on your computer.
3.  **Install Dependencies:** Open a command prompt or PowerShell, navigate to the directory where you saved the script, and run the following commands to install the necessary Python libraries:
    ```bash
    pip install psutil
    pip install py-nvml-dev
    pip install pywin32
    ```

## Running the Monitor

1.  Navigate to the directory containing `system_monitor.py` in a command prompt or PowerShell.
2.  Execute the script using:
    ```bash
    python system_monitor.py
    ```
3.  The script will start logging information to `system_monitor.log` in the same directory.
4.  To stop the monitor, press `Ctrl+C` in the command prompt window.

## Log File

*   The log file is named `system_monitor.log` and is created in the same directory as the script.
*   It contains timestamped entries for:
    *   CPU usage and temperature.
    *   RAM usage.
    *   GPU temperature, utilization, and fan speed (for NVIDIA GPUs).
    *   Critical errors from the Windows "System" and "Application" event logs.

## Configuration and Long-Term Use

*   **Polling Interval:** The data collection interval can be adjusted by changing the `POLLING_INTERVAL` constant at the top of the `system_monitor.py` script.
*   **Log Rotation:** The log file will grow indefinitely. For long-term monitoring, it is recommended to implement log rotation. See comments near the `logging.basicConfig()` line in the script for suggestions (e.g., using `RotatingFileHandler`).

## Disclaimer

This tool provides information to help diagnose issues. It does not guarantee a solution for any specific hardware or software problem. For NVIDIA GPU monitoring, ensure you have the appropriate NVIDIA drivers installed. CPU temperature monitoring is system-dependent.
