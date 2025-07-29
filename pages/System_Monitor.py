import streamlit as st
import time

# Import resource monitoring
try:
    from resource_monitor import (
        start_resource_monitoring,
        stop_resource_monitoring,
        get_resource_stats,
        create_resource_dashboard,
        get_resource_summary,
        export_resource_data,
        resource_monitor,
    )

    RESOURCE_MONITORING_AVAILABLE = True
except ImportError:
    RESOURCE_MONITORING_AVAILABLE = False
    st.warning("Resource monitoring not available. Install psutil: pip install psutil")


def main():
    st.set_page_config(page_title="System Monitor", page_icon="📊", layout="wide")

    st.title("📊 System Resource Monitor")
    st.markdown("Real-time monitoring of system resources and application metrics.")

    # Initialize resource monitoring
    if RESOURCE_MONITORING_AVAILABLE:
        if "resource_monitoring_started" not in st.session_state:
            start_resource_monitoring()
            st.session_state.resource_monitoring_started = True

        # Current stats with loading state
        with st.spinner("📊 Loading resource statistics..."):
            current_stats = get_resource_stats()

        if "error" not in current_stats:
            # System Metrics
            st.subheader("🖥️ System Metrics")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("CPU Usage", f"{current_stats['cpu_percent']:.1f}%")
                st.metric("Memory Usage", f"{current_stats['memory_percent']:.1f}%")

            with col2:
                st.metric("Memory (GB)", f"{current_stats['memory_gb']:.2f} GB")
                st.metric("Disk Usage", f"{current_stats['disk_usage_percent']:.1f}%")

            with col3:
                st.metric("Network Sent", f"{current_stats['network_sent_mb']:.1f} MB")
                st.metric("Network Recv", f"{current_stats['network_recv_mb']:.1f} MB")

            with col4:
                st.metric("Process Count", current_stats["process_count"])
                st.metric("Uptime", f"{current_stats['uptime_seconds']/60:.1f} min")

            # Application-specific metrics
            st.subheader("📈 Application Metrics")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("YFinance Calls", current_stats["yfinance_calls"])

            with col2:
                st.metric(
                    "Prophet Training Time",
                    f"{current_stats['prophet_training_time']:.2f}s",
                )

            with col3:
                st.metric("Streamlit Requests", current_stats["streamlit_requests"])

            # Summary statistics
            summary_stats = get_resource_summary()
            if summary_stats:
                st.subheader("📋 Summary Statistics")
                col1, col2 = st.columns(2)

                with col1:
                    st.write(
                        f"**Average CPU Usage:** {summary_stats.get('avg_cpu_percent', 0):.1f}%"
                    )
                    st.write(
                        f"**Max CPU Usage:** {summary_stats.get('max_cpu_percent', 0):.1f}%"
                    )
                    st.write(
                        f"**Average Memory Usage:** {summary_stats.get('avg_memory_percent', 0):.1f}%"
                    )
                    st.write(
                        f"**Max Memory Usage:** {summary_stats.get('max_memory_percent', 0):.1f}%"
                    )

                with col2:
                    st.write(
                        f"**Total Network Sent:** {summary_stats.get('total_network_sent_mb', 0):.1f} MB"
                    )
                    st.write(
                        f"**Total Network Recv:** {summary_stats.get('total_network_recv_mb', 0):.1f} MB"
                    )
                    st.write(
                        f"**Total Uptime:** {summary_stats.get('total_uptime_minutes', 0):.1f} minutes"
                    )
                    st.write(
                        f"**Total YFinance Calls:** {summary_stats.get('yfinance_calls', 0)}"
                    )

            # Export data button
            if st.button("📥 Export Resource Data"):
                filename = export_resource_data()
                st.success(f"Resource data exported to: {filename}")

        else:
            st.error(f"Error getting resource stats: {current_stats['error']}")
    else:
        st.warning(
            "Resource monitoring is not available. Please install psutil: pip install psutil"
        )


if __name__ == "__main__":
    main()
