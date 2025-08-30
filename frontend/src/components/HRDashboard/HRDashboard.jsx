import React, { useState, useEffect } from 'react';
import { apiService } from '../../services/api';
import ChartRenderer from '../ChartRenderer/ChartRenderer';

const HRDashboard = ({ isVisible = false }) => {
  const [employees, setEmployees] = useState([]);
  const [departments, setDepartments] = useState([]);
  const [attendanceStats, setAttendanceStats] = useState(null);
  const [tasks, setTasks] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (isVisible) {
      loadDashboardData();
    }
  }, [isVisible]);

  const loadDashboardData = async () => {
    setLoading(true);
    setError(null);
    
    try {
      // Load basic HR data
      const [employeesData, departmentsData, tasksData] = await Promise.all([
        apiService.getEmployees().catch(err => {
          console.warn('Failed to load employees:', err);
          return [];
        }),
        apiService.getDepartments().catch(err => {
          console.warn('Failed to load departments:', err);
          return [];
        }),
        apiService.getTasks().catch(err => {
          console.warn('Failed to load tasks:', err);
          return [];
        })
      ]);

      setEmployees(employeesData);
      setDepartments(departmentsData);
      setTasks(tasksData);

      // Load attendance statistics
      try {
        const attendanceData = await apiService.getAttendanceStatistics({ days: 7 });
        setAttendanceStats(attendanceData);
      } catch (err) {
        console.warn('Failed to load attendance statistics:', err);
      }

    } catch (err) {
      setError('Failed to load dashboard data. Please check if the backend server is running.');
      console.error('Dashboard data loading error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleRefresh = () => {
    loadDashboardData();
  };

  if (!isVisible) return null;

  if (loading) {
    return (
      <div className="hr-dashboard">
        <div className="dashboard-loading">
          <div className="loading-dots">
            <span></span>
            <span></span>
            <span></span>
          </div>
          <p>Loading HR Dashboard...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="hr-dashboard">
        <div className="dashboard-error">
          <p>{error}</p>
          <button onClick={handleRefresh} className="refresh-button">
            Try Again
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="hr-dashboard">
      <div className="dashboard-header">
        <h2>HR Dashboard</h2>
        <button onClick={handleRefresh} className="refresh-button">
          Refresh
        </button>
      </div>

      <div className="dashboard-grid">
        {/* Employee Summary */}
        <div className="dashboard-card">
          <h3>Employee Summary</h3>
          <div className="summary-stats">
            <div className="stat-item">
              <span className="stat-number">{employees.length}</span>
              <span className="stat-label">Total Employees</span>
            </div>
            <div className="stat-item">
              <span className="stat-number">{departments.length}</span>
              <span className="stat-label">Departments</span>
            </div>
          </div>
          
          {departments.length > 0 && (
            <div className="department-list">
              <h4>Departments:</h4>
              <ul>
                {departments.map(dept => (
                  <li key={dept.id}>{dept.name}</li>
                ))}
              </ul>
            </div>
          )}
        </div>

        {/* Recent Tasks */}
        <div className="dashboard-card">
          <h3>Recent Tasks</h3>
          {tasks.length > 0 ? (
            <div className="task-list">
              {tasks.slice(0, 5).map((task, index) => (
                <div key={task.id || index} className="task-item">
                  <div className="task-content">{task.task_description || task.description}</div>
                  <div className="task-meta">
                    {task.employee_name && <span>By: {task.employee_name}</span>}
                    {task.department_name && <span>Dept: {task.department_name}</span>}
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <p>No recent tasks found.</p>
          )}
        </div>

        {/* Attendance Statistics */}
        {attendanceStats && attendanceStats.chart_data && (
          <div className="dashboard-card chart-card">
            <ChartRenderer chartData={attendanceStats.chart_data} />
          </div>
        )}
      </div>
    </div>
  );
};

export default HRDashboard;