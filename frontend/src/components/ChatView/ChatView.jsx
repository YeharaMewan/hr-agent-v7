import React, { useState, useRef, useEffect } from 'react';
import MessageBubble from '../MessageBubble/MessageBubble';
import SuggestionChips from '../SuggestionChips/SuggestionChips';
import EmployeeFormModal from '../EmployeeFormModal/EmployeeFormModal';
import { apiService } from '../../services/api';

const ChatView = ({ messages, setMessages }) => {
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [chipsHidden, setChipsHidden] = useState(false);
  const [isEmployeeFormOpen, setIsEmployeeFormOpen] = useState(false);
  const [employeeFormData, setEmployeeFormData] = useState(null);
  const [employeeFormAction, setEmployeeFormAction] = useState('');
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);
  const chatContainerRef = useRef(null);
  const [isMobile, setIsMobile] = useState(false);
  const [isKeyboardOpen, setIsKeyboardOpen] = useState(false);
  const [originalViewportHeight, setOriginalViewportHeight] = useState(window.innerHeight);

  // Enhanced employee management intent detection
  const detectEmployeeManagementIntent = (message) => {
    const messageLower = message.toLowerCase();
    
    const creationTriggers = [
      'add employee', 'create employee', 'new employee', 'add new employee', 
      'add user', 'create user', 'new user', 'add new user',
      'need to add employee', 'can you add employee', 'help me add employee'
    ];
    
    const updateTriggers = [
      'update employee', 'change employee', 'modify employee', 'edit employee',
      'update user', 'change user', 'edit user', 'modify user',
      'employee details', 'user details', 'update details',
      // Enhanced patterns for specific name updates
      'update .+? details', 'change .+? information', 'edit .+? details',
      'modify .+? details', '.+? details', 'update .+?', 'edit .+?'
    ];
    
    const lookupTriggers = [
      'find employee', 'show employee', 'get employee', 'employee information',
      'find user', 'show user', 'get user', 'user information'
    ];

    const isCreation = creationTriggers.some(trigger => messageLower.includes(trigger));
    
    // Enhanced update detection with regex patterns
    const isUpdate = updateTriggers.some(trigger => {
      if (trigger.includes('.+?')) {
        // Use regex for patterns with wildcards
        const regexPattern = trigger.replace(/\.\+\?/g, '[\\w\\s]+');
        const regex = new RegExp(regexPattern, 'i');
        return regex.test(message);
      }
      return messageLower.includes(trigger);
    });
    
    const isLookup = lookupTriggers.some(trigger => messageLower.includes(trigger));
    
    if (isCreation) return { type: 'create', action: 'create_employee' };
    if (isUpdate) return { type: 'update', action: 'update_employee' };
    if (isLookup) return { type: 'lookup', action: 'query_employee' };
    
    return null;
  };

  // Fallback form triggering when backend doesn't send form request
  const triggerFallbackEmployeeForm = (intentData, originalMessage) => {
    console.log('Triggering fallback employee form for:', intentData);
    
    // Extract potential employee name from the message
    const extractNameFromMessage = (message) => {
      const messageLower = message.toLowerCase();
      
      // Patterns to extract names from messages
      const namePatterns = [
        /(?:add|create|new|update|change|modify|edit|find|show)\s+(?:employee|user)?\s+([A-Za-z]{2,}\s+[A-Za-z]{2,}(?:\s+[A-Za-z]+)?)/i,
        /(?:employee|user)\s+([A-Za-z]{2,}\s+[A-Za-z]{2,}(?:\s+[A-Za-z]+)?)/i,
        /([A-Za-z]{2,}\s+[A-Za-z]{2,}(?:\s+[A-Za-z]+)?)\s*(?:details|information|info)/i
      ];
      
      for (const pattern of namePatterns) {
        const match = message.match(pattern);
        if (match) {
          const potentialName = match[1].trim();
          // Filter out common non-name words
          const excludeWords = ['employee', 'user', 'details', 'information', 'new', 'add', 'create'];
          if (!excludeWords.some(word => potentialName.toLowerCase().includes(word))) {
            return potentialName;
          }
        }
      }
      return '';
    };

    const extractedName = extractNameFromMessage(originalMessage);
    
    // Create default form data based on intent type
    let defaultFormData = {
      departments: [
        { value: 1, name: "AI", label: "AI Department" },
        { value: 2, name: "Marketing", label: "Marketing Department" },
        { value: 3, name: "HR", label: "HR Department" },
        { value: 4, name: "IT", label: "IT Department" },
        { value: 5, name: "Construction", label: "Construction Department" },
        { value: 6, name: "Finance", label: "Finance Department" }
      ],
      roles: [
        { value: "hr", label: "HR Staff" },
        { value: "leader", label: "Team Leader" },
        { value: "employee", label: "Employee" }
      ]
    };

    if (intentData.type === 'create') {
      defaultFormData = {
        ...defaultFormData,
        pre_filled: {
          name: extractedName,
          email: '',
          role: '',
          department: '',
          phone_number: '',
          address: ''
        }
      };
      
      setEmployeeFormData(defaultFormData);
      setEmployeeFormAction('create_employee');
      setIsEmployeeFormOpen(true);
      
      // No message - just show the form popup directly
      
    } else if (intentData.type === 'update' && extractedName) {
      // For updates, try to fetch employee data first, then show form
      fetchEmployeeDataAndShowForm(extractedName, originalMessage);
    } else if (intentData.type === 'update' && !extractedName) {
      // Generic update request without specific name
      defaultFormData = {
        ...defaultFormData,
        current_values: {
          name: '',
          email: '',
          role: '',
          department: '',
          phone_number: '',
          address: ''
        },
        employee_id: null
      };
      
      setEmployeeFormData(defaultFormData);
      setEmployeeFormAction('update_employee');
      setIsEmployeeFormOpen(true);
      
      // No message - just show the form popup directly
    }
  };

  // Fetch employee data and show update form with pre-filled information
  const fetchEmployeeDataAndShowForm = async (employeeName, originalMessage) => {
    try {
      // No loading message - just fetch data and show form

      // Try to fetch employee data by searching
      const searchResponse = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: `show me ${employeeName} details` }),
      });

      if (searchResponse.ok) {
        // Parse the response to extract employee data
        // This is a simplified approach - in a real implementation, 
        // you might want a dedicated API endpoint for employee lookup
        
        // For now, show the form with the extracted name
        const defaultFormData = {
          departments: [
            { value: 1, name: "AI", label: "AI Department" },
            { value: 2, name: "Marketing", label: "Marketing Department" },
            { value: 3, name: "HR", label: "HR Department" },
            { value: 4, name: "IT", label: "IT Department" },
            { value: 5, name: "Construction", label: "Construction Department" },
            { value: 6, name: "Finance", label: "Finance Department" }
          ],
          roles: [
            { value: "hr", label: "HR Staff" },
            { value: "leader", label: "Team Leader" },
            { value: "employee", label: "Employee" }
          ],
          current_values: {
            name: employeeName,
            email: '',
            role: '',
            department: '',
            phone_number: '',
            address: ''
          },
          employee_id: null // Will be resolved when form is submitted
        };
        
        setEmployeeFormData(defaultFormData);
        setEmployeeFormAction('update_employee');
        setIsEmployeeFormOpen(true);
        
        // No success message - just show the form
      } else {
        throw new Error('Failed to fetch employee data');
      }
    } catch (error) {
      console.error('Error fetching employee data:', error);
      
      // Show form anyway with just the name
      const defaultFormData = {
        departments: [
          { value: 1, name: "AI", label: "AI Department" },
          { value: 2, name: "Marketing", label: "Marketing Department" },
          { value: 3, name: "HR", label: "HR Department" },
          { value: 4, name: "IT", label: "IT Department" },
          { value: 5, name: "Construction", label: "Construction Department" },
          { value: 6, name: "Finance", label: "Finance Department" }
        ],
        roles: [
          { value: "hr", label: "HR Staff" },
          { value: "leader", label: "Team Leader" },
          { value: "employee", label: "Employee" }
        ],
        current_values: {
          name: employeeName,
          email: '',
          role: '',
          department: '',
          phone_number: '',
          address: ''
        },
        employee_id: null
      };
      
      setEmployeeFormData(defaultFormData);
      setEmployeeFormAction('update_employee');
      setIsEmployeeFormOpen(true);
      
      // No error message - just show the form even if lookup failed
    }
  };

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Focus input when component mounts
  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  // Mobile detection and keyboard handling
  useEffect(() => {
    const checkIfMobile = () => {
      const userAgent = navigator.userAgent || navigator.vendor || window.opera;
      const isMobileDevice = /android|iphone|ipad|ipod|blackberry|iemobile|opera mini/i.test(userAgent.toLowerCase());
      const isTouchDevice = 'ontouchstart' in window || navigator.maxTouchPoints > 0;
      const isSmallScreen = window.innerWidth <= 768;
      
      return isMobileDevice || (isTouchDevice && isSmallScreen);
    };

    setIsMobile(checkIfMobile());
    setOriginalViewportHeight(window.visualViewport?.height || window.innerHeight);
  }, []);

  // Enhanced keyboard visibility detection for mobile with dynamic positioning
  useEffect(() => {
    if (!isMobile) return;

    const handleViewportChange = () => {
      const viewport = window.visualViewport || window;
      const currentHeight = viewport.height || window.innerHeight;
      const heightDifference = originalViewportHeight - currentHeight;
      const keyboardThreshold = 150; // Minimum height difference to consider keyboard open
      
      const keyboardIsOpen = heightDifference > keyboardThreshold;
      
      if (keyboardIsOpen !== isKeyboardOpen) {
        setIsKeyboardOpen(keyboardIsOpen);
        
        // Calculate exact keyboard height and available space
        const keyboardHeight = Math.max(0, heightDifference);
        const availableHeight = currentHeight;
        const inputBottomOffset = keyboardHeight > keyboardThreshold ? Math.max(60, keyboardHeight * 0.08) : 0; // Much higher above keyboard
        
        // Update CSS custom properties for dynamic positioning
        if (document.documentElement) {
          document.documentElement.style.setProperty('--keyboard-height', `${keyboardHeight}px`);
          document.documentElement.style.setProperty('--available-height', `${availableHeight}px`);
          document.documentElement.style.setProperty('--input-bottom-offset', `${inputBottomOffset}px`);
        }
        
        // Enhanced positioning when keyboard opens
        if (keyboardIsOpen && inputRef.current) {
          // Multiple positioning strategies for better compatibility
          const positionInputAboveKeyboard = () => {
            // Strategy 1: Use visual viewport positioning
            const inputRect = inputRef.current?.getBoundingClientRect();
            if (inputRect && viewport.height) {
              // Calculate position to keep input visible above keyboard
              const desiredInputBottom = viewport.height - 20; // 20px from top of available area
              const currentInputBottom = inputRect.bottom;
              
              if (currentInputBottom > desiredInputBottom) {
                const scrollAdjustment = currentInputBottom - desiredInputBottom;
                window.scrollBy({
                  top: scrollAdjustment,
                  behavior: 'smooth'
                });
              }
            }
            
            // Strategy 2: Ensure input container is positioned correctly
            const inputContainer = document.querySelector('.chat-container.mobile.keyboard-open .input-area-container');
            if (inputContainer) {
              // Force re-calculation of fixed positioning with higher offset
              inputContainer.style.bottom = `${inputBottomOffset}px`;
              // Also scroll the page up to ensure full visibility
              setTimeout(() => {
                window.scrollBy({
                  top: -50, // Scroll up a bit to ensure full chatbox visibility
                  behavior: 'smooth'
                });
              }, 150);
            }
          };
          
          // Execute positioning with appropriate delays
          setTimeout(positionInputAboveKeyboard, 100);
          
          // Additional positioning check after keyboard animation
          setTimeout(() => {
            if (inputRef.current && document.activeElement === inputRef.current) {
              inputRef.current.scrollIntoView({ 
                behavior: 'smooth', 
                block: 'nearest',
                inline: 'nearest'
              });
            }
          }, 300);
          
          // Add visual feedback for keyboard state
          if (chatContainerRef.current) {
            chatContainerRef.current.style.transition = 'transform 0.3s ease';
            chatContainerRef.current.style.transform = 'translateZ(0)';
          }
        } else if (!keyboardIsOpen) {
          // Reset positioning when keyboard closes
          if (document.documentElement) {
            document.documentElement.style.setProperty('--keyboard-height', '0px');
            document.documentElement.style.setProperty('--available-height', '100vh');
            document.documentElement.style.setProperty('--input-bottom-offset', '0px');
          }
          
          if (chatContainerRef.current) {
            chatContainerRef.current.style.transform = '';
          }
        }
      }
    };

    // Use visualViewport API if available (modern browsers)
    if (window.visualViewport) {
      window.visualViewport.addEventListener('resize', handleViewportChange);
      return () => {
        window.visualViewport.removeEventListener('resize', handleViewportChange);
      };
    } else {
      // Fallback for older browsers
      window.addEventListener('resize', handleViewportChange);
      return () => {
        window.removeEventListener('resize', handleViewportChange);
      };
    }
  }, [isMobile, isKeyboardOpen, originalViewportHeight]);

  // Enhanced mobile input focus with cross-device keyboard positioning
  const handleMobileInputFocus = () => {
    if (isMobile && inputRef.current) {
      // Small delay to ensure proper focusing
      setTimeout(() => {
        inputRef.current?.focus();
        // Prevent zoom on iOS
        inputRef.current?.setAttribute('readonly', 'readonly');
        inputRef.current?.removeAttribute('readonly');
        
        // Cross-device positioning strategies
        const positionInputAboveKeyboard = () => {
          const viewport = window.visualViewport || window;
          const inputElement = inputRef.current;
          const chatContainer = chatContainerRef.current;
          
          if (!inputElement || !chatContainer) return;
          
          // Device-specific positioning strategies
          const userAgent = navigator.userAgent.toLowerCase();
          const isIOS = /iphone|ipad|ipod/.test(userAgent);
          const isAndroid = /android/.test(userAgent);
          
          // Get current dimensions
          const inputRect = inputElement.getBoundingClientRect();
          const currentHeight = viewport.height || window.innerHeight;
          const keyboardHeight = originalViewportHeight - currentHeight;
          
          if (keyboardHeight > 150) {
            // iOS-specific handling
            if (isIOS) {
              // iOS needs higher positioning to be fully visible
              const inputBottomOffset = Math.max(50, keyboardHeight * 0.15);
              document.documentElement.style.setProperty('--input-bottom-offset', `${inputBottomOffset}px`);
              
              // Smooth scroll to ensure visibility
              setTimeout(() => {
                inputElement.scrollIntoView({ 
                  behavior: 'smooth', 
                  block: 'start', // Changed to start for higher positioning
                  inline: 'nearest'
                });
              }, 150);
            }
            // Android-specific handling
            else if (isAndroid) {
              // Android needs much more aggressive positioning to be visible
              const inputBottomOffset = Math.max(80, keyboardHeight * 0.18);
              document.documentElement.style.setProperty('--input-bottom-offset', `${inputBottomOffset}px`);
              
              // Force container repositioning for Android
              const inputContainer = document.querySelector('.chat-container.mobile.keyboard-open .input-area-container');
              if (inputContainer) {
                inputContainer.style.bottom = `${inputBottomOffset}px`;
                inputContainer.style.position = 'fixed';
              }
              
              // Additional scroll adjustment - scroll page up more
              setTimeout(() => {
                const newInputRect = inputElement.getBoundingClientRect();
                if (newInputRect.bottom > currentHeight - 100) {
                  window.scrollBy({
                    top: newInputRect.bottom - currentHeight + 120, // Increased scroll amount
                    behavior: 'smooth'
                  });
                }
              }, 200);
            }
            // Fallback for other devices
            else {
              const inputBottomOffset = Math.max(70, keyboardHeight * 0.15);
              document.documentElement.style.setProperty('--input-bottom-offset', `${inputBottomOffset}px`);
              
              // Standard scrollIntoView with higher positioning
              setTimeout(() => {
                inputElement.scrollIntoView({ 
                  behavior: 'smooth', 
                  block: 'start', // Changed to start for higher positioning
                  inline: 'nearest'
                });
              }, 100);
            }
          }
        };
        
        // Execute positioning with device-appropriate delays
        const userAgent = navigator.userAgent.toLowerCase();
        const delay = /iphone|ipad|ipod/.test(userAgent) ? 300 : 200;
        setTimeout(positionInputAboveKeyboard, delay);
      }, 100);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (inputValue.trim() && !isLoading) {
      // Enhanced keyboard hiding on mobile after sending message
      if (isMobile && inputRef.current) {
        const hideKeyboard = () => {
          // Primary method: blur the input
          inputRef.current.blur();
          
          // Secondary method: temporarily remove focus and disable input
          inputRef.current.setAttribute('readonly', 'readonly');
          inputRef.current.disabled = true;
          
          // Re-enable after a short delay
          setTimeout(() => {
            if (inputRef.current) {
              inputRef.current.removeAttribute('readonly');
              inputRef.current.disabled = false;
            }
          }, 100);
          
          // Force layout reflow to ensure keyboard dismissal
          document.body.scrollTop = document.body.scrollTop;
          
          // Additional fallback: scroll to slightly change viewport
          if (window.scrollY > 0) {
            window.scrollTo({
              top: window.scrollY - 1,
              behavior: 'instant'
            });
            setTimeout(() => {
              window.scrollTo({
                top: window.scrollY + 1,
                behavior: 'instant'
              });
            }, 50);
          }
        };
        
        hideKeyboard();
        
        // Additional check after delay to ensure keyboard is hidden
        setTimeout(() => {
          if (inputRef.current && document.activeElement === inputRef.current) {
            inputRef.current.blur();
            // Move focus to body as fallback
            if (document.body.focus) {
              document.body.focus();
            }
          }
        }, 200);
      }
      await sendMessage(inputValue);
    }
  };

  const sendMessage = async (messageText) => {
    const userMessage = {
      id: Date.now(),
      content: messageText,
      isUser: true,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    // Enhanced fallback detection for employee management queries
    const shouldTriggerEmployeeForm = detectEmployeeManagementIntent(messageText);
    
    // Add loading message
    const loadingMessage = {
      id: Date.now() + 1,
      content: '',
      isUser: false,
      isLoading: true,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, loadingMessage]);

    try {
      // Use the API service for chat messages
      const response = await apiService.sendChatMessage(messageText);

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      // Track if we received a form request from backend
      let formRequestReceived = false;

      // Handle streaming response using API service handler
      await apiService.handleStreamingResponse(
        response,
        (data) => {
          // Handle each chunk of data
          if (data.type === 'chart') {
            updateMessageWithChart(loadingMessage.id, data.payload);
          } else if (data.type === 'text') {
            updateMessageWithText(loadingMessage.id, data.content);
          } else if (data.type === 'form_request') {
            formRequestReceived = true;
            // Handle form request - show employee form modal
            handleFormRequest(data);
            // Remove the loading message instead of updating it with text
            setMessages(prev => prev.filter(msg => msg.id !== loadingMessage.id));
          } else if (data.type === 'employee_not_found_create_offer') {
            formRequestReceived = true;
            // Handle employee not found - show create offer and potentially open form
            handleEmployeeNotFoundOffer(data);
            // Remove the loading message instead of showing not found text
            setMessages(prev => prev.filter(msg => msg.id !== loadingMessage.id));
          } else if (data.type === 'human_loop_question') {
            // Handle human loop question - show message with follow-up options
            handleHumanLoopQuestion(data);
            // Update the loading message with the question content
            updateMessageWithText(loadingMessage.id, data.content);
          }
        },
        () => {
          // Handle completion - check for fallback form triggering
          setMessages(prev => prev.map(msg => 
            msg.id === loadingMessage.id 
              ? { ...msg, isLoading: false }
              : msg
          ));
          
          // Fallback: If we detected employee management intent but no form was triggered
          if (shouldTriggerEmployeeForm && !formRequestReceived) {
            console.log('Fallback form trigger detected for:', shouldTriggerEmployeeForm);
            triggerFallbackEmployeeForm(shouldTriggerEmployeeForm, messageText);
          }
        },
        (error) => {
          // Handle error
          console.error('Streaming error:', error);
          setMessages(prev => prev.map(msg => 
            msg.id === loadingMessage.id 
              ? {
                  ...msg,
                  content: 'Error occurred while receiving response',
                  isLoading: false,
                  isError: true
                }
              : msg
          ));
        }
      );

    } catch (error) {
      console.error('Error sending message:', error);
      
      // Remove loading message and add error message
      setMessages(prev => {
        const filtered = prev.filter(msg => msg.id !== loadingMessage.id);
        return [...filtered, {
          id: Date.now() + 2,
          content: 'Sorry, an error occurred while generating the response. Please check if the backend server is running.',
          isUser: false,
          isError: true,
          timestamp: new Date()
        }];
      });
    } finally {
      setIsLoading(false);
    }
  };

  // Helper functions for updating messages with streaming data
  const updateMessageWithText = (messageId, textContent) => {
    setMessages(prev => prev.map(msg => {
      if (msg.id === messageId) {
        return {
          ...msg,
          content: (msg.content || '') + textContent,
          isLoading: false
        };
      }
      return msg;
    }));
  };

  const updateMessageWithChart = (messageId, chartPayload) => {
    setMessages(prev => prev.map(msg => {
      if (msg.id === messageId) {
        return {
          ...msg,
          content: '',
          isLoading: false,
          hasChart: true,
          chartData: chartPayload
        };
      }
      return msg;
    }));
  };

  // Employee form handling functions
  const handleFormRequest = (data) => {
    console.log('Form request received:', data);
    
    // Extract form data from the response (backend sends form_data directly now)
    const formData = data.form_data || data.payload?.form_data || {};
    const action = data.action || data.payload?.action || 'create_employee';
    
    console.log('Extracted form data:', formData);
    console.log('Action:', action);
    
    setEmployeeFormData(formData);
    setEmployeeFormAction(action);
    setIsEmployeeFormOpen(true);
  };

  const handleEmployeeNotFoundOffer = (data) => {
    console.log('Employee not found offer:', data);
    
    // Could show a different UI for this case, but for now we'll just show the message
    // In the future, could add a button to trigger form creation
  };

  const handleHumanLoopQuestion = (data) => {
    console.log('Human loop question received:', data);
    
    // Store conversation state if provided
    if (data.conversation_state) {
      console.log('Storing conversation state:', data.conversation_state);
      // In production, this would be stored in a more persistent way
      window.conversationState = data.conversation_state;
    }
    
    // The message content will be displayed automatically by updateMessageWithText
    // Follow-up questions will be displayed as suggestion chips if provided
    if (data.follow_up_questions && data.follow_up_questions.length > 0) {
      // Add follow-up questions as clickable suggestions
      setTimeout(() => {
        console.log('Adding follow-up suggestions:', data.follow_up_questions);
        // This could be enhanced to show special follow-up buttons
      }, 100);
    }
  };

  const handleFormSubmitSuccess = (response) => {
    console.log('Form submitted successfully:', response);
    
    // Enhanced success message based on action type
    const actionType = employeeFormAction === 'create_employee' ? 'created' : 'updated';
    const employeeName = employeeFormData?.current_values?.name || employeeFormData?.pre_filled?.name || 'Employee';
    
    const successMessage = {
      id: Date.now(),
      content: `✅ **Success!** ${employeeName} has been ${actionType} successfully!\n\n${response.message || `Employee record ${actionType} with all the provided information.`}`,
      isUser: false,
      timestamp: new Date(),
      isSuccess: true
    };
    
    setMessages(prev => [...prev, successMessage]);
    
    // Close the form
    setIsEmployeeFormOpen(false);
    
    // Clear form data
    setEmployeeFormData(null);
    setEmployeeFormAction('');
  };

  const handleFormSubmitError = (error) => {
    console.error('Form submission error:', error);
    
    // Enhanced error message with more context
    const actionType = employeeFormAction === 'create_employee' ? 'creating' : 'updating';
    const employeeName = employeeFormData?.current_values?.name || employeeFormData?.pre_filled?.name || 'employee';
    
    const errorMessage = {
      id: Date.now(),
      content: `❌ **Error ${actionType} ${employeeName}**\n\n${error.message || `An error occurred while ${actionType} the employee record. Please check the information and try again.`}\n\nYou can modify the form and try again.`,
      isUser: false,
      timestamp: new Date(),
      isError: true
    };
    
    setMessages(prev => [...prev, errorMessage]);
  };

  const handleFormClose = () => {
    setIsEmployeeFormOpen(false);
    setEmployeeFormData(null);
    setEmployeeFormAction('');
  };


  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const handleChipClick = (command) => {
    sendMessage(command);
  };

  const toggleChips = () => {
    setChipsHidden(!chipsHidden);
  };

  // Enhanced attendance-focused suggestion chips
  const chatSuggestionChips = [
    // Quick Attendance Actions
    { emoji: '✅', text: 'Mark Present', command: 'Mark my attendance as Present for today' },
    { emoji: '🏠', text: 'Work From Home', command: 'Mark my attendance as Work from home for today' },
    { emoji: '📅', text: 'Plan Leave', command: 'Mark my attendance as Planned Leave for tomorrow' },
    { emoji: '🏥', text: 'Medical Leave', command: 'Mark my attendance as Medical Leave for today' },
    
    // Attendance Reports & Analytics
    { emoji: '📊', text: 'My Summary', command: 'Show my attendance summary for the last 30 days with insights' },
    { emoji: '📈', text: 'Department Trends', command: 'Give me attendance trends analysis for all departments in the last 2 weeks' },
    { emoji: '🔍', text: 'Attendance Insights', command: 'Provide attendance insights and pattern analysis for the last 30 days' },
    { emoji: '⚖️', text: 'Compare Teams', command: 'Generate a comparative attendance report for different departments' },
    
    // Advanced Operations
    { emoji: '📝', text: 'Update Yesterday', command: 'Update my attendance for yesterday to Work from home' },
    { emoji: '✔️', text: 'Validate Data', command: 'Validate attendance data for Thavindu with status Present for today' },
    { emoji: '👥', text: 'Team Report', command: 'Show detailed attendance report for Construction department last 7 days' },
    { emoji: '🎯', text: 'Employee Lookup', command: 'Find all employees in the IT department and their recent attendance' },
    
    // Statistics & Visualization
    { emoji: '🍰', text: 'Pie Chart Stats', command: 'Show attendance statistics as a pie chart for the last 14 days' },
    { emoji: '📋', text: 'Summary Report', command: 'Generate summary attendance report for all employees last week' },
    { emoji: '📊', text: 'Weekly Trends', command: 'Show weekly attendance trends with day-of-week analysis' },
    { emoji: '🏢', text: 'Org Overview', command: 'Give me organization-wide attendance insights and recommendations' },
  ];

  const LoadingDots = () => (
    <div className="loading-dots">
      <span></span>
      <span></span>
      <span></span>
    </div>
  );

  return (
    <div 
      ref={chatContainerRef}
      className={`chat-container ${isMobile ? 'mobile' : ''} ${isKeyboardOpen ? 'keyboard-open' : ''}`}
    >
      {/* Header */}
      <div className="chat-header">
        <img src="/RiseHRLogo.png" alt="RiseHR Logo" className="header-logo" />
      </div>

      {/* Messages */}
      <div className="chat-messages">
        <div className="message-content-wrapper">
          {messages.map((message) => (
            <div key={message.id}>
              {message.isLoading ? (
                <div className="message-wrapper bot">
                  <div className="message-bubble">
                    <LoadingDots />
                  </div>
                </div>
              ) : (
                <MessageBubble message={message} />
              )}
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Input Area */}
      <div className="input-area-container">
        <div className="input-content-wrapper">
          <SuggestionChips
            chips={chatSuggestionChips}
            onChipClick={handleChipClick}
            isLanding={false}
            isHidden={chipsHidden}
          />
          
          <form onSubmit={handleSubmit} className="chat-input-wrapper">
            <button
              type="button"
              className="chip-toggle-button"
              onClick={toggleChips}
              title="Toggle Suggestions"
            >
              💡
            </button>
            
            <input
              ref={inputRef}
              type="text"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyPress={handleKeyPress}
              onFocus={handleMobileInputFocus}
              onTouchStart={handleMobileInputFocus}
              className="chat-input"
              placeholder="Ask a follow-up..."
              autoComplete="off"
              disabled={isLoading}
              inputMode={isMobile ? "text" : undefined}
            />
            
            <button
              type="submit"
              className="send-button"
              disabled={!inputValue.trim() || isLoading}
              title="Send message"
              aria-label="Send message"
            >
              <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="m3 3 3 9-3 9 19-9Z"/>
                <path d="m6 12 16 0"/>
              </svg>
            </button>
          </form>
        </div>
      </div>

      {/* Employee Form Modal */}
      <EmployeeFormModal
        isOpen={isEmployeeFormOpen}
        onClose={handleFormClose}
        formData={employeeFormData}
        action={employeeFormAction}
        onSubmitSuccess={handleFormSubmitSuccess}
        onSubmitError={handleFormSubmitError}
      />
    </div>
  );
};

export default ChatView;