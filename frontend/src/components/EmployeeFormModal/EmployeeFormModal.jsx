import React, { useState, useEffect } from 'react';
import { apiService } from '../../services/api';
import './EmployeeFormModal.css';

const EmployeeFormModal = ({ 
  isOpen, 
  onClose, 
  formData, 
  action, 
  onSubmitSuccess,
  onSubmitError 
}) => {
  const [formValues, setFormValues] = useState({
    name: '',
    email: '',
    role: '',
    department: '',
    phone_number: '',
    address: ''
  });
  
  const [errors, setErrors] = useState({});
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [departments, setDepartments] = useState([]);
  const [roles, setRoles] = useState([]);

  // Initialize form when modal opens or formData changes
  useEffect(() => {
    if (isOpen && formData) {
      // Set form values from formData
      const preFilledValues = formData.pre_filled || {};
      const currentValues = formData.current_values || {};
      
      console.log('Initializing form with data:', { preFilledValues, currentValues, formData });
      
      setFormValues({
        name: preFilledValues.name || currentValues.name || '',
        email: preFilledValues.email || currentValues.email || '',
        role: preFilledValues.role || currentValues.role || '',
        department: preFilledValues.department || currentValues.department || '',
        phone_number: preFilledValues.phone_number || currentValues.phone_number || '',
        address: preFilledValues.address || currentValues.address || ''
      });
      
      // Set dropdown options
      setDepartments(formData.departments || []);
      setRoles(formData.roles || []);
      setErrors({});
    }
  }, [isOpen, formData]);

  const handleInputChange = (field, value) => {
    setFormValues(prev => ({
      ...prev,
      [field]: value
    }));
    
    // Clear error for this field when user starts typing
    if (errors[field]) {
      setErrors(prev => ({
        ...prev,
        [field]: ''
      }));
    }
  };

  const validateForm = () => {
    const newErrors = {};
    
    // Required field validation
    if (!formValues.name.trim()) {
      newErrors.name = 'Name is required';
    }
    
    if (!formValues.email.trim()) {
      newErrors.email = 'Email is required';
    } else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(formValues.email)) {
      newErrors.email = 'Please enter a valid email address';
    }
    
    if (!formValues.role) {
      newErrors.role = 'Role is required';
    }
    
    if (!formValues.department) {
      newErrors.department = 'Department is required';
    }
    
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!validateForm()) {
      return;
    }
    
    setIsSubmitting(true);
    
    try {
      let response;
      
      if (action === 'create_employee') {
        // Call create employee endpoint
        response = await apiService.createEmployee({
          query: `Create employee ${formValues.name}`,
          name: formValues.name,
          email: formValues.email,
          role: formValues.role,
          department: formValues.department,
          phone_number: formValues.phone_number,
          address: formValues.address
        });
      } else if (action === 'update_employee') {
        // Call update employee endpoint
        const employeeId = formData.employee_id;
        response = await apiService.updateEmployee({
          query: `Update employee ${employeeId}`,
          employee_identifier: employeeId.toString(),
          field_updates: {
            name: formValues.name,
            email: formValues.email,
            role: formValues.role,
            department: formValues.department,
            phone_number: formValues.phone_number,
            address: formValues.address
          },
          update_reason: "Updated via form"
        });
      }
      
      if (response.success) {
        onSubmitSuccess(response);
        handleClose();
      } else {
        onSubmitError(response);
      }
      
    } catch (error) {
      console.error('Form submission error:', error);
      onSubmitError({
        success: false,
        message: 'An error occurred while submitting the form. Please try again.',
        error: error.message
      });
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleClose = () => {
    setFormValues({
      name: '',
      email: '',
      role: '',
      department: '',
      phone_number: '',
      address: ''
    });
    setErrors({});
    setIsSubmitting(false);
    onClose();
  };

  const getDepartmentValue = () => {
    if (!formValues.department) return '';
    
    // If department is already an ID, return it
    if (typeof formValues.department === 'number' || /^\d+$/.test(formValues.department)) {
      return formValues.department.toString();
    }
    
    // Find department by name (check both 'name' and 'label' fields)
    const dept = departments.find(d => 
      d.name?.toLowerCase() === formValues.department.toLowerCase() ||
      d.label?.toLowerCase() === formValues.department.toLowerCase()
    );
    return dept ? dept.value.toString() : '';
  };

  if (!isOpen) return null;

  return (
    <div className="modal-overlay" onClick={handleClose}>
      <div className="employee-form-modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h2>
            {action === 'create_employee' ? 'Create New Employee' : 'Update Employee'}
          </h2>
          <button className="modal-close-btn" onClick={handleClose}>
            ✕
          </button>
        </div>

        <form onSubmit={handleSubmit} className="employee-form">
          <div className="form-row">
            <div className="form-field">
              <label htmlFor="name">Full Name *</label>
              <input
                id="name"
                type="text"
                value={formValues.name}
                onChange={(e) => handleInputChange('name', e.target.value)}
                className={errors.name ? 'error' : ''}
                placeholder="Enter full name"
                disabled={isSubmitting}
              />
              {errors.name && <span className="error-message">{errors.name}</span>}
            </div>

            <div className="form-field">
              <label htmlFor="email">Email Address *</label>
              <input
                id="email"
                type="email"
                value={formValues.email}
                onChange={(e) => handleInputChange('email', e.target.value)}
                className={errors.email ? 'error' : ''}
                placeholder="Enter email address"
                disabled={isSubmitting}
              />
              {errors.email && <span className="error-message">{errors.email}</span>}
            </div>
          </div>

          <div className="form-row">
            <div className="form-field">
              <label htmlFor="role">Role *</label>
              <select
                id="role"
                value={formValues.role}
                onChange={(e) => handleInputChange('role', e.target.value)}
                className={errors.role ? 'error' : ''}
                disabled={isSubmitting}
              >
                <option value="">Select a role</option>
                {roles.map((role) => (
                  <option key={role.value} value={role.value}>
                    {role.label}
                  </option>
                ))}
              </select>
              {errors.role && <span className="error-message">{errors.role}</span>}
            </div>

            <div className="form-field">
              <label htmlFor="department">Department *</label>
              <select
                id="department"
                value={getDepartmentValue()}
                onChange={(e) => {
                  const selectedDept = departments.find(d => d.value.toString() === e.target.value);
                  // Store the department name for backend compatibility
                  handleInputChange('department', selectedDept ? selectedDept.name || selectedDept.label : e.target.value);
                }}
                className={errors.department ? 'error' : ''}
                disabled={isSubmitting}
              >
                <option value="">Select a department</option>
                {departments.map((dept) => (
                  <option key={dept.value} value={dept.value}>
                    {dept.label}
                  </option>
                ))}
              </select>
              {errors.department && <span className="error-message">{errors.department}</span>}
            </div>
          </div>

          <div className="form-row">
            <div className="form-field">
              <label htmlFor="phone_number">Phone Number</label>
              <input
                id="phone_number"
                type="tel"
                value={formValues.phone_number}
                onChange={(e) => handleInputChange('phone_number', e.target.value)}
                placeholder="Enter phone number"
                disabled={isSubmitting}
              />
            </div>
          </div>

          <div className="form-field">
            <label htmlFor="address">Address</label>
            <textarea
              id="address"
              value={formValues.address}
              onChange={(e) => handleInputChange('address', e.target.value)}
              placeholder="Enter address"
              rows="3"
              disabled={isSubmitting}
            />
          </div>

          <div className="form-actions">
            <button
              type="button"
              onClick={handleClose}
              className="btn btn-secondary"
              disabled={isSubmitting}
            >
              Cancel
            </button>
            <button
              type="submit"
              className="btn btn-primary"
              disabled={isSubmitting}
            >
              {isSubmitting ? (
                <>
                  <span className="spinner"></span>
                  {action === 'create_employee' ? 'Creating...' : 'Updating...'}
                </>
              ) : (
                action === 'create_employee' ? 'Create Employee' : 'Update Employee'
              )}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default EmployeeFormModal;