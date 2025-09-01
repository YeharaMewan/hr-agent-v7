// API service layer for HR Management System backend integration
const API_BASE = import.meta.env.VITE_API_URL || import.meta.env.VITE_API_BASE || 'http://localhost:8000';

class ApiService {
  constructor() {
    this.baseURL = API_BASE;
  }

  /**
   * Send a message to the chat endpoint
   * @param {string} message - The user's message
   * @returns {Promise<Response>} Streaming response
   */
  async sendChatMessage(message) {
    try {
      const response = await fetch(`${this.baseURL}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return response;
    } catch (error) {
      console.error('Chat API Error:', error);
      throw error;
    }
  }

  /**
   * Send a streaming query to the /chat endpoint (alias for compatibility)
   * @param {string} query - The user's query
   * @returns {Promise<Response>} Streaming response
   */
  async sendStreamingQuery(query) {
    return this.sendChatMessage(query);
  }

  /**
   * Get system health status
   * @returns {Promise<Object>} Health status
   */
  async getHealthStatus() {
    try {
      const response = await fetch(`${this.baseURL}/health`);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Health Check Error:', error);
      throw error;
    }
  }

  /**
   * Get system info and capabilities
   * @returns {Promise<Object>} System info
   */
  async getSystemInfo() {
    try {
      const response = await fetch(`${this.baseURL}/`);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('System Info Error:', error);
      throw error;
    }
  }

  // HR-specific API methods

  /**
   * Get all employees with optional filtering
   * @param {Object} filters - Filter options (name, department, role, is_active)
   * @returns {Promise<Array>} List of employees
   */
  async getEmployees(filters = {}) {
    try {
      const params = new URLSearchParams();
      Object.entries(filters).forEach(([key, value]) => {
        if (value !== null && value !== undefined && value !== '') {
          params.append(key, value);
        }
      });
      
      const url = params.toString() ? `${this.baseURL}/employees?${params}` : `${this.baseURL}/employees`;
      const response = await fetch(url);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Get Employees Error:', error);
      throw error;
    }
  }

  /**
   * Get all departments
   * @returns {Promise<Array>} List of departments
   */
  async getDepartments() {
    try {
      const response = await fetch(`${this.baseURL}/departments`);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Get Departments Error:', error);
      throw error;
    }
  }

  /**
   * Mark attendance for an employee
   * @param {Object} attendanceData - Attendance data (employee_id, status, notes)
   * @returns {Promise<Object>} Response from server
   */
  async markAttendance(attendanceData) {
    try {
      const response = await fetch(`${this.baseURL}/attendance/mark`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(attendanceData),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Mark Attendance Error:', error);
      throw error;
    }
  }

  /**
   * Get attendance records with filtering
   * @param {Object} filters - Filter options (employee_name, department, start_date, end_date, status, days)
   * @returns {Promise<Array>} List of attendance records
   */
  async getAttendanceRecords(filters = {}) {
    try {
      const params = new URLSearchParams();
      Object.entries(filters).forEach(([key, value]) => {
        if (value !== null && value !== undefined && value !== '') {
          params.append(key, value);
        }
      });
      
      const url = params.toString() ? `${this.baseURL}/attendance?${params}` : `${this.baseURL}/attendance`;
      const response = await fetch(url);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Get Attendance Records Error:', error);
      throw error;
    }
  }

  /**
   * Get attendance statistics
   * @param {Object} filters - Filter options (days, department, role)
   * @returns {Promise<Object>} Attendance statistics
   */
  async getAttendanceStatistics(filters = {}) {
    try {
      const params = new URLSearchParams();
      Object.entries(filters).forEach(([key, value]) => {
        if (value !== null && value !== undefined && value !== '') {
          params.append(key, value);
        }
      });
      
      const url = params.toString() ? `${this.baseURL}/attendance/statistics?${params}` : `${this.baseURL}/attendance/statistics`;
      const response = await fetch(url);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Get Attendance Statistics Error:', error);
      throw error;
    }
  }

  /**
   * Get tasks with filtering
   * @param {Object} filters - Filter options (leader_name, department, priority, location)
   * @returns {Promise<Array>} List of tasks
   */
  async getTasks(filters = {}) {
    try {
      const params = new URLSearchParams();
      Object.entries(filters).forEach(([key, value]) => {
        if (value !== null && value !== undefined && value !== '') {
          params.append(key, value);
        }
      });
      
      const url = params.toString() ? `${this.baseURL}/tasks?${params}` : `${this.baseURL}/tasks`;
      const response = await fetch(url);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Get Tasks Error:', error);
      throw error;
    }
  }

  // Employee Management API methods

  /**
   * Create a new employee via intelligent agent
   * @param {Object} employeeData - Employee creation data (query, name, email, role, department, phone_number, address)
   * @returns {Promise<Object>} Response from server
   */
  async createEmployee(employeeData) {
    try {
      const response = await fetch(`${this.baseURL}/tools/create_employee_intelligent`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(employeeData),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Create Employee Error:', error);
      throw error;
    }
  }

  /**
   * Update an employee via intelligent agent
   * @param {Object} updateData - Employee update data (query, employee_identifier, field_updates, update_reason)
   * @returns {Promise<Object>} Response from server
   */
  async updateEmployee(updateData) {
    try {
      const response = await fetch(`${this.baseURL}/tools/update_employee_intelligent`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(updateData),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Update Employee Error:', error);
      throw error;
    }
  }

  /**
   * Search for employees in management context
   * @param {string} query - Search query (name, email, ID, or natural language)
   * @returns {Promise<Object>} Search results with suggestions
   */
  async searchEmployeeForManagement(query) {
    try {
      const response = await fetch(`${this.baseURL}/tools/search_employee_for_management`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Search Employee Error:', error);
      throw error;
    }
  }

  /**
   * Get form data for frontend (departments, roles, validation rules)
   * @returns {Promise<Object>} Form configuration data
   */
  async getFormData() {
    try {
      const response = await fetch(`${this.baseURL}/tools/get_form_data_for_frontend`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({}),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Get Form Data Error:', error);
      throw error;
    }
  }

  /**
   * Parse Server-Sent Events from streaming response
   * @param {Response} response - The streaming response
   * @param {Function} onChunk - Callback for each data chunk
   * @param {Function} onComplete - Callback when streaming is complete
   * @param {Function} onError - Callback for errors
   */
  async handleStreamingResponse(response, onChunk, onComplete, onError) {
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let lineBuffer = '';

    try {
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        lineBuffer += decoder.decode(value, { stream: true });
        const lines = lineBuffer.split('\n\n');
        lineBuffer = lines.pop();

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;

          const jsonData = line.substring(6).trim();
          if (!jsonData || jsonData === '[DONE]') continue;

          try {
            const data = JSON.parse(jsonData);
            
            if (data.type === 'complete') {
              onComplete?.();
              return;
            } else if (data.type === 'error') {
              onError?.(new Error(data.error || 'Streaming error'));
              return;
            } else {
              onChunk?.(data);
            }
          } catch (e) {
            console.error('Error parsing JSON chunk:', jsonData, e);
          }
        }
      }
      
      onComplete?.();
    } catch (error) {
      console.error('Streaming processing error:', error);
      onError?.(error);
    }
  }
}

// Create and export a singleton instance
export const apiService = new ApiService();

// Export the class for testing purposes
export { ApiService };

// Helper function to check if backend is available
export async function checkBackendStatus() {
  try {
    const health = await apiService.getHealthStatus();
    return {
      available: true,
      status: health.status,
      details: health
    };
  } catch (error) {
    return {
      available: false,
      error: error.message,
      details: null
    };
  }
}