import psutil
import time
import threading
import plotly.graph_objects as go
from datetime import datetime
import json


class ResourceMonitor:
    """Monitor system resources for the QueryStockAI."""

    def __init__(self):
        self.monitoring = False
        self.monitor_thread = None
        self.resource_data = {
            "timestamps": [],
            "cpu_percent": [],
            "memory_percent": [],
            "memory_mb": [],
            "disk_usage_percent": [],
            "network_sent_mb": [],
            "network_recv_mb": [],
            "process_count": [],
            "yfinance_calls": 0,
            "prophet_training_time": 0,
            "streamlit_requests": 0,
        }
        self.start_time = None
        self.process = psutil.Process()

    def start_monitoring(self):
        """Start resource monitoring in a separate thread."""
        if not self.monitoring:
            self.monitoring = True
            self.start_time = datetime.now()
            self.monitor_thread = threading.Thread(
                target=self._monitor_loop, daemon=True
            )
            self.monitor_thread.start()
            return True
        return False

    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)

    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                # Get current timestamp
                timestamp = datetime.now()

                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)

                # Memory usage
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                memory_mb = memory.used / (1024 * 1024)  # Convert to MB

                # Disk usage
                disk = psutil.disk_usage("/")
                disk_usage_percent = disk.percent

                # Network usage
                network = psutil.net_io_counters()
                network_sent_mb = network.bytes_sent / (1024 * 1024)
                network_recv_mb = network.bytes_recv / (1024 * 1024)

                # Process count
                process_count = len(psutil.pids())

                # Store data
                self.resource_data["timestamps"].append(timestamp)
                self.resource_data["cpu_percent"].append(cpu_percent)
                self.resource_data["memory_percent"].append(memory_percent)
                self.resource_data["memory_mb"].append(memory_mb)
                self.resource_data["disk_usage_percent"].append(disk_usage_percent)
                self.resource_data["network_sent_mb"].append(network_sent_mb)
                self.resource_data["network_recv_mb"].append(network_recv_mb)
                self.resource_data["process_count"].append(process_count)

                # Keep only last 1000 data points to prevent memory issues
                max_points = 1000
                if len(self.resource_data["timestamps"]) > max_points:
                    for key in self.resource_data:
                        if isinstance(self.resource_data[key], list):
                            self.resource_data[key] = self.resource_data[key][
                                -max_points:
                            ]

                time.sleep(2)  # Monitor every 2 seconds

            except Exception as e:
                print(f"Error in monitoring loop: {e}")
                time.sleep(5)

    def get_current_stats(self) -> Dict:
        """Get current resource statistics."""
        try:
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")
            network = psutil.net_io_counters()

            return {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": memory.percent,
                "memory_mb": memory.used / (1024 * 1024),
                "memory_gb": memory.used / (1024 * 1024 * 1024),
                "disk_usage_percent": disk.percent,
                "disk_free_gb": disk.free / (1024 * 1024 * 1024),
                "network_sent_mb": network.bytes_sent / (1024 * 1024),
                "network_recv_mb": network.bytes_recv / (1024 * 1024),
                "process_count": len(psutil.pids()),
                "uptime_seconds": (
                    (datetime.now() - self.start_time).total_seconds()
                    if self.start_time
                    else 0
                ),
                "yfinance_calls": self.resource_data["yfinance_calls"],
                "prophet_training_time": self.resource_data["prophet_training_time"],
                "streamlit_requests": self.resource_data["streamlit_requests"],
            }
        except Exception as e:
            return {"error": str(e)}

    def increment_yfinance_calls(self):
        """Increment yfinance API call counter."""
        self.resource_data["yfinance_calls"] += 1

    def add_prophet_training_time(self, seconds: float):
        """Add Prophet training time."""
        self.resource_data["prophet_training_time"] += seconds

    def increment_streamlit_requests(self):
        """Increment Streamlit request counter."""
        self.resource_data["streamlit_requests"] += 1

    def create_resource_dashboard(self) -> go.Figure:
        """Create a comprehensive resource dashboard."""
        if not self.resource_data["timestamps"]:
            return None

        # Create subplots
        fig = go.Figure()

        # CPU Usage
        fig.add_trace(
            go.Scatter(
                x=self.resource_data["timestamps"],
                y=self.resource_data["cpu_percent"],
                mode="lines",
                name="CPU %",
                line=dict(color="red", width=2),
            )
        )

        # Memory Usage
        fig.add_trace(
            go.Scatter(
                x=self.resource_data["timestamps"],
                y=self.resource_data["memory_percent"],
                mode="lines",
                name="Memory %",
                line=dict(color="blue", width=2),
                yaxis="y2",
            )
        )

        # Memory Usage in MB
        fig.add_trace(
            go.Scatter(
                x=self.resource_data["timestamps"],
                y=self.resource_data["memory_mb"],
                mode="lines",
                name="Memory (MB)",
                line=dict(color="lightblue", width=2),
                yaxis="y3",
            )
        )

        # Network Usage
        fig.add_trace(
            go.Scatter(
                x=self.resource_data["timestamps"],
                y=self.resource_data["network_sent_mb"],
                mode="lines",
                name="Network Sent (MB)",
                line=dict(color="green", width=2),
                yaxis="y4",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=self.resource_data["timestamps"],
                y=self.resource_data["network_recv_mb"],
                mode="lines",
                name="Network Recv (MB)",
                line=dict(color="orange", width=2),
                yaxis="y4",
            )
        )

        # Update layout
        fig.update_layout(
            title="System Resource Usage",
            xaxis_title="Time",
            height=600,
            hovermode="x unified",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
            yaxis=dict(title="CPU %", side="left"),
            yaxis2=dict(title="Memory %", side="right", overlaying="y"),
            yaxis3=dict(title="Memory (MB)", side="right", position=0.95),
            yaxis4=dict(title="Network (MB)", side="right", position=0.9),
        )

        return fig

    def get_summary_stats(self) -> Dict:
        """Get summary statistics."""
        if not self.resource_data["timestamps"]:
            return {}

        return {
            "total_uptime_minutes": (
                (datetime.now() - self.start_time).total_seconds() / 60
                if self.start_time
                else 0
            ),
            "avg_cpu_percent": sum(self.resource_data["cpu_percent"])
            / len(self.resource_data["cpu_percent"]),
            "max_cpu_percent": max(self.resource_data["cpu_percent"]),
            "avg_memory_percent": sum(self.resource_data["memory_percent"])
            / len(self.resource_data["memory_percent"]),
            "max_memory_percent": max(self.resource_data["memory_percent"]),
            "avg_memory_mb": sum(self.resource_data["memory_mb"])
            / len(self.resource_data["memory_mb"]),
            "max_memory_mb": max(self.resource_data["memory_mb"]),
            "total_network_sent_mb": sum(self.resource_data["network_sent_mb"]),
            "total_network_recv_mb": sum(self.resource_data["network_recv_mb"]),
            "yfinance_calls": self.resource_data["yfinance_calls"],
            "prophet_training_time": self.resource_data["prophet_training_time"],
            "streamlit_requests": self.resource_data["streamlit_requests"],
        }

    def export_data(self, filename: str = None):
        """Export monitoring data to JSON file."""
        if filename is None:
            filename = (
                f"resource_monitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )

        export_data = {
            "summary_stats": self.get_summary_stats(),
            "monitoring_data": {
                "timestamps": [
                    ts.isoformat() for ts in self.resource_data["timestamps"]
                ],
                "cpu_percent": self.resource_data["cpu_percent"],
                "memory_percent": self.resource_data["memory_percent"],
                "memory_mb": self.resource_data["memory_mb"],
                "disk_usage_percent": self.resource_data["disk_usage_percent"],
                "network_sent_mb": self.resource_data["network_sent_mb"],
                "network_recv_mb": self.resource_data["network_recv_mb"],
                "process_count": self.resource_data["process_count"],
            },
        }

        with open(filename, "w") as f:
            json.dump(export_data, f, indent=2)

        return filename


# Global monitor instance
resource_monitor = ResourceMonitor()


def start_resource_monitoring():
    """Start resource monitoring."""
    return resource_monitor.start_monitoring()


def stop_resource_monitoring():
    """Stop resource monitoring."""
    resource_monitor.stop_monitoring()


def get_resource_stats():
    """Get current resource statistics."""
    return resource_monitor.get_current_stats()


def create_resource_dashboard():
    """Create resource dashboard."""
    return resource_monitor.create_resource_dashboard()


def get_resource_summary():
    """Get resource summary."""
    return resource_monitor.get_summary_stats()


def export_resource_data(filename=None):
    """Export resource data."""
    return resource_monitor.export_data(filename)
