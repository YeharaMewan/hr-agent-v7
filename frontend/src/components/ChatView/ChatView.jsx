import React, { useState, useRef, useEffect } from 'react';
import MessageBubble from '../MessageBubble/MessageBubble';
import SuggestionChips from '../SuggestionChips/SuggestionChips';
import { apiService } from '../../services/api';

const ChatView = ({ messages, setMessages }) => {
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [chipsHidden, setChipsHidden] = useState(false);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Focus input when component mounts
  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (inputValue.trim() && !isLoading) {
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

      // Handle streaming response using API service handler
      await apiService.handleStreamingResponse(
        response,
        (data) => {
          // Handle each chunk of data
          if (data.type === 'chart') {
            updateMessageWithChart(loadingMessage.id, data.payload);
          } else if (data.type === 'text') {
            updateMessageWithText(loadingMessage.id, data.content);
          }
        },
        () => {
          // Handle completion
          setMessages(prev => prev.map(msg => 
            msg.id === loadingMessage.id 
              ? { ...msg, isLoading: false }
              : msg
          ));
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
    <div className="chat-container">
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
              className="chat-input"
              placeholder="Ask a follow-up..."
              autoComplete="off"
              disabled={isLoading}
            />
            
            <button
              type="submit"
              className="send-button"
              disabled={!inputValue.trim() || isLoading}
            >
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor">
                <path d="M3.478 2.405a.75.75 0 00-.926.94l2.432 7.905H13.5a.75.75 0 010 1.5H4.984l-2.432 7.905a.75.75 0 00.926.94 60.519 60.519 0 0018.445-8.986.75.75 0 000-1.218A60.517 60.517 0 003.478 2.405z" />
              </svg>
            </button>
          </form>
        </div>
      </div>
    </div>
  );
};

export default ChatView;