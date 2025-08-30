import React, { useEffect, useRef } from 'react';
import { Chart, registerables } from 'chart.js';

// Register all Chart.js components
Chart.register(...registerables);

const ChartRenderer = ({ chartData }) => {
  const canvasRef = useRef(null);
  const chartRef = useRef(null);

  useEffect(() => {
    if (!chartData || !canvasRef.current) return;

    // Destroy existing chart if it exists
    if (chartRef.current) {
      chartRef.current.destroy();
    }

    try {
      const ctx = canvasRef.current.getContext('2d');
      
      // Configure chart options for better responsive behavior
      const options = {
        ...chartData.chart_config?.options,
        maintainAspectRatio: false,
        responsive: true,
      };

      chartRef.current = new Chart(ctx, {
        ...chartData.chart_config,
        options,
      });
    } catch (error) {
      console.error('Error creating chart:', error);
    }

    // Cleanup function
    return () => {
      if (chartRef.current) {
        chartRef.current.destroy();
      }
    };
  }, [chartData]);

  if (!chartData) {
    return <div>No chart data available</div>;
  }

  return (
    <div className="interactive-chart-container">
      {chartData.title && (
        <div className="chart-title">
          <h3>{chartData.title}</h3>
        </div>
      )}
      
      {/* Chart filters could be added here if needed */}
      {chartData.filter_options && chartData.filter_options.departments && (
        <div className="chart-filters">
          <label>Department: </label>
          <select onChange={(e) => {
            // Handle department filter change - can be implemented later
            console.log('Department filter changed:', e.target.value);
          }}>
            {chartData.filter_options.departments.map((dept, index) => (
              <option key={index} value={dept}>{dept}</option>
            ))}
          </select>
        </div>
      )}

      <div className="chart-canvas-wrapper" style={{ height: '300px' }}>
        <canvas ref={canvasRef}></canvas>
      </div>
    </div>
  );
};

export default ChartRenderer;