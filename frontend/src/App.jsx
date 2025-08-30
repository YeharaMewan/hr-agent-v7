import React, { useState } from 'react';
import LandingView from './components/LandingView/LandingView';
import ChatView from './components/ChatView/ChatView';
import { apiService } from './services/api';
import './styles/globals.css';

function App() {
  const [currentView, setCurrentView] = useState('landing');
  const [messages, setMessages] = useState([]);


  const switchToChatView = () => {
    setCurrentView('chat');
  };

  const handleFirstMessage = async (messageText) => {
    if (!messageText.trim()) return;
    
    switchToChatView();
    
    // Add user message to chat
    const userMessage = {
      id: Date.now(),
      content: messageText,
      isUser: true,
      timestamp: new Date()
    };
    
    setMessages(prev => [...prev, userMessage]);
    
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

  const addMessage = (message) => {
    setMessages(prev => [...prev, message]);
  };

  return (
    <div className="view-container">

      {/* Landing View */}
      <div className={`view ${currentView === 'chat' ? 'hidden' : ''}`}>
        <LandingView onFirstMessage={handleFirstMessage} />
      </div>

      {/* Chat View */}
      <div className={`view ${currentView === 'landing' ? 'hidden' : ''}`}>
        <ChatView 
          messages={messages} 
          setMessages={setMessages}
          addMessage={addMessage}
        />
      </div>
    </div>
  );
}

export default App;
