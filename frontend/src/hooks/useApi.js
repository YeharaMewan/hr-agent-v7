import { useState, useCallback } from 'react';
import { apiService } from '../services/api';

/**
 * Custom hook for handling API calls with loading and error states
 * @param {Object} options - Hook configuration options
 * @returns {Object} API state and methods
 */
export function useApi(options = {}) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [data, setData] = useState(null);

  const { onSuccess, onError } = options;

  const execute = useCallback(async (apiCall) => {
    setLoading(true);
    setError(null);
    
    try {
      const result = await apiCall();
      setData(result);
      onSuccess?.(result);
      return result;
    } catch (err) {
      setError(err);
      onError?.(err);
      throw err;
    } finally {
      setLoading(false);
    }
  }, [onSuccess, onError]);

  const sendQuery = useCallback((query) => {
    return execute(() => apiService.sendQuery(query));
  }, [execute]);

  const getHealthStatus = useCallback(() => {
    return execute(() => apiService.getHealthStatus());
  }, [execute]);

  const getCapabilities = useCallback(() => {
    return execute(() => apiService.getCapabilities());
  }, [execute]);

  const reset = useCallback(() => {
    setLoading(false);
    setError(null);
    setData(null);
  }, []);

  return {
    loading,
    error,
    data,
    sendQuery,
    getHealthStatus,
    getCapabilities,
    execute,
    reset
  };
}