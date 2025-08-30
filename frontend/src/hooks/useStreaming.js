import { useState, useCallback } from 'react';
import { apiService } from '../services/api';

/**
 * Custom hook for handling streaming API responses
 * @param {Object} options - Hook configuration options
 * @returns {Object} Streaming state and methods
 */
export function useStreaming(options = {}) {
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState(null);
  const [accumulatedData, setAccumulatedData] = useState('');
  const [chartData, setChartData] = useState(null);

  const { onChunk, onComplete, onError } = options;

  const sendStreamingQuery = useCallback(async (query) => {
    setIsStreaming(true);
    setError(null);
    setAccumulatedData('');
    setChartData(null);

    try {
      const response = await apiService.sendStreamingQuery(query);

      await apiService.handleStreamingResponse(
        response,
        (chunk) => {
          // Handle different chunk types
          if (chunk.type === 'text') {
            setAccumulatedData(prev => prev + chunk.content);
          } else if (chunk.type === 'chart') {
            setChartData(chunk.payload);
            setAccumulatedData(''); // Clear text for chart messages
          }
          
          onChunk?.(chunk);
        },
        () => {
          setIsStreaming(false);
          onComplete?.();
        },
        (err) => {
          setError(err);
          setIsStreaming(false);
          onError?.(err);
        }
      );
    } catch (err) {
      setError(err);
      setIsStreaming(false);
      onError?.(err);
    }
  }, [onChunk, onComplete, onError]);

  const reset = useCallback(() => {
    setIsStreaming(false);
    setError(null);
    setAccumulatedData('');
    setChartData(null);
  }, []);

  return {
    isStreaming,
    error,
    accumulatedData,
    chartData,
    sendStreamingQuery,
    reset
  };
}

/**
 * Custom hook for managing chat messages with streaming support
 * @returns {Object} Messages state and methods
 */
export function useChatMessages() {
  const [messages, setMessages] = useState([]);

  const addMessage = useCallback((message) => {
    setMessages(prev => [...prev, {
      id: Date.now() + Math.random(),
      timestamp: new Date(),
      ...message
    }]);
  }, []);

  const addUserMessage = useCallback((content) => {
    addMessage({
      content,
      isUser: true
    });
  }, [addMessage]);

  const addBotMessage = useCallback((content, options = {}) => {
    addMessage({
      content,
      isUser: false,
      ...options
    });
  }, [addMessage]);

  const updateMessage = useCallback((messageId, updates) => {
    setMessages(prev => prev.map(msg => 
      msg.id === messageId ? { ...msg, ...updates } : msg
    ));
  }, []);

  const removeMessage = useCallback((messageId) => {
    setMessages(prev => prev.filter(msg => msg.id !== messageId));
  }, []);

  const clearMessages = useCallback(() => {
    setMessages([]);
  }, []);

  return {
    messages,
    setMessages,
    addMessage,
    addUserMessage,
    addBotMessage,
    updateMessage,
    removeMessage,
    clearMessages
  };
}