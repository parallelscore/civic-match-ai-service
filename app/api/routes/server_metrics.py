import time

import psutil
from fastapi import APIRouter, status
from fastapi.responses import HTMLResponse

from app.utils.logging_util import setup_logger


class ServerMetrics:

    def __init__(self, app):
        self.app = app
        self.router = APIRouter()
        self.logger = setup_logger(__name__)
        self.router.add_api_route('/', self.server_metrics, methods=['GET'],
                                  status_code=status.HTTP_200_OK)

    async def server_metrics(self) -> HTMLResponse:
        current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())

        # Get the server uptime
        uptime_seconds = time.time() - self.app.state.start_time
        uptime = str(int(uptime_seconds // 3600)) + ":" + \
                 str(int((uptime_seconds % 3600) // 60)) + \
                 ":" + str(int(uptime_seconds % 60))

        # Get the current memory usage
        memory_info = psutil.virtual_memory()
        total_memory_mb = memory_info.total / (1024 ** 2)
        available_memory_mb = memory_info.available / (1024 ** 2)
        used_memory_mb = (memory_info.total - memory_info.available) / (1024 ** 2)
        free_memory_mb = memory_info.free / (1024 ** 2)
        percent_memory_used = memory_info.percent

        # Determine memory color
        if percent_memory_used < 50:
            memory_color = 'green'
        elif percent_memory_used < 75:
            memory_color = 'orange'
        else:
            memory_color = 'red'

        # Get swap memory usage
        swap_info = psutil.swap_memory()
        total_swap_mb = swap_info.total / (1024 ** 2)
        used_swap_mb = swap_info.used / (1024 ** 2)
        free_swap_mb = swap_info.free / (1024 ** 2)
        percent_swap_used = swap_info.percent

        # Determine swap color
        if percent_swap_used < 50:
            swap_color = 'green'
        elif percent_swap_used < 75:
            swap_color = 'orange'
        else:
            swap_color = 'red'

        # Get the current CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_percents = psutil.cpu_percent(interval=1, percpu=True)

        # Determine CPU color
        if cpu_percent < 50:
            cpu_color = 'green'
        elif cpu_percent < 75:
            cpu_color = 'orange'
        else:
            cpu_color = 'red'

        # Get the disk usage
        disk_usage = psutil.disk_usage('/')
        total_disk_gb = disk_usage.total / (1024 ** 3)
        used_disk_gb = disk_usage.used / (1024 ** 3)
        free_disk_gb = disk_usage.free / (1024 ** 3)
        percent_disk_used = disk_usage.percent

        # Determine disk color
        if percent_disk_used < 50:
            disk_color = 'green'
        elif percent_disk_used < 75:
            disk_color = 'orange'
        else:
            disk_color = 'red'

        # Get disk I/O stats
        disk_io = psutil.disk_io_counters()
        read_bytes_mb = disk_io.read_bytes / (1024 ** 2)
        write_bytes_mb = disk_io.write_bytes / (1024 ** 2)

        # Get network I/O stats
        net_io = psutil.net_io_counters()
        bytes_sent_mb = net_io.bytes_sent / (1024 ** 2)
        bytes_recv_mb = net_io.bytes_recv / (1024 ** 2)

        # Get the load average
        load_avg = psutil.getloadavg()

        # Get the number of active processes
        num_processes = len(psutil.pids())

        html_content = f"""
            <html>
                <head>
                    <title>Server Metrics</title>
                    <style>
                        body {{
                            font-family: Arial, sans-serif;
                            margin: 0;
                            padding: 20px;
                            background-color: #f0f0f0;
                        }}
                        h1 {{
                            color: #333;
                        }}
                        .metrics {{
                            background-color: #fff;
                            padding: 20px;
                            border-radius: 8px;
                            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                        }}
                        .metrics ul {{
                            list-style-type: none;
                            padding: 0;
                        }}
                        .metrics li {{
                            margin: 5px 0;
                        }}
                        .metrics strong {{
                            color: #333;
                        }}
                        .metrics span {{
                            color: #007bff;
                        }}
                    </style>
                </head>
                <body>
                    <h1>Server Metrics</h1>
                    <div class="metrics">
                        <p><strong>Current Server Time:</strong> <span>{current_time}</span></p>
                        <p><strong>Server Uptime:</strong> <span>{uptime}</span></p>
                        <p><strong>Requests Processed:</strong> <span>{self.app.state.requests_processed}</span></p>
                        <p><strong>Memory Info:</strong>
                            <ul>
                                <li>Total: <span>{total_memory_mb:.2f} MB</span></li>
                                <li>Available: <span>{available_memory_mb:.2f} MB</span></li>
                                <li>Percent Used: <span style="color:{memory_color}">{percent_memory_used:.2f}%</span
                                ></li> <li>Used: <span>{used_memory_mb:.2f} MB</span></li>
                                <li>Free: <span>{free_memory_mb:.2f} MB</span></li>
                            </ul>
                        </p>
                        <p><strong>Swap Memory Info:</strong>
                            <ul>
                                <li>Total: <span>{total_swap_mb:.2f} MB</span></li>
                                <li>Used: <span>{used_swap_mb:.2f} MB</span></li>
                                <li>Free: <span>{free_swap_mb:.2f} MB</span></li>
                                <li>Percent Used: <span style="color:{swap_color}">{percent_swap_used:.2f}%</span></li>
                            </ul>
                        </p>
                        <p><strong>CPU Usage:</strong> <span style="color:{cpu_color}">{cpu_percent:.2f}%</span></p>
                        <p><strong>CPU Usage per Core:</strong>
                            <ul>
                                {''.join([f'''<li>Core {i + 1}: <span style="color:{"green" if cpu < 50 else "orange"
        if cpu < 75 else "red"}">{cpu:.2f}%</span></li>''' for i, cpu in enumerate(cpu_percents)])}
                            </ul>
                        </p>
                        <p><strong>Disk Usage:</strong>
                            <ul>
                                <li>Total: <span>{total_disk_gb:.2f} GB</span></li>
                                <li>Used: <span>{used_disk_gb:.2f} GB</span></li>
                                <li>Free: <span>{free_disk_gb:.2f} GB</span></li>
                                <li>Percent Used: <span style="color:{disk_color}">{percent_disk_used:.2f}%</span></li>
                            </ul>
                        </p>
                        <p><strong>Disk I/O:</strong>
                            <ul>
                                <li>Read: <span>{read_bytes_mb:.2f} MB</span></li>
                                <li>Write: <span>{write_bytes_mb:.2f} MB</span></li>
                            </ul>
                        </p>
                        <p><strong>Network I/O:</strong>
                            <ul>
                                <li>Bytes Sent: <span>{bytes_sent_mb:.2f} MB</span></li>
                                <li>Bytes Received: <span>{bytes_recv_mb:.2f} MB</span></li>
                            </ul>
                        </p>
                        <p><strong>Load Average:</strong>
                            <ul>
                                <li>1 min: <span>{load_avg[0]:.2f}</span></li>
                                <li>5 min: <span>{load_avg[1]:.2f}</span></li>
                                <li>15 min: <span>{load_avg[2]:.2f}</span></li>
                            </ul>
                        </p>
                        <p><strong>Number of Active Processes:</strong> <span>{num_processes}</span></p>
                    </div>
                </body>
            </html>
            """

        return HTMLResponse(content=html_content)
