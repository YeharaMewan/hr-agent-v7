import React, { useState } from 'react';
import { apiService } from '../../services/api';
import './DownloadButton.css';

const DownloadButton = ({ documentInfo, onDownloadStart, onDownloadComplete, onError }) => {
  const [isDownloading, setIsDownloading] = useState(false);
  const [downloadError, setDownloadError] = useState(null);
  const [downloadSuccess, setDownloadSuccess] = useState(null);



  // Simple download function - downloads to browser's default folder
  const handleDownload = async () => {
    if (isDownloading || !documentInfo) return;
    
    try {
      setIsDownloading(true);
      setDownloadError(null);
      setDownloadSuccess(null);
      
      if (onDownloadStart) {
        onDownloadStart(documentInfo);
      }
      
      // Enhanced validation for document ID
      if (!documentInfo.document_id) {
        throw new Error('Document ID is missing. Please generate a new document.');
      }
      
      if (documentInfo.document_id.startsWith('temp-')) {
        throw new Error('Document is not ready for download yet. Please wait for generation to complete or try generating again.');
      }
      
      // Validate UUID format
      const uuidPattern = /^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$/i;
      if (!uuidPattern.test(documentInfo.document_id)) {
        throw new Error('Invalid document ID format. Please generate a new document.');
      }
      
      // Create filename
      const filename = documentInfo.filename || `${documentInfo.document_type}_${documentInfo.employee_name}.pdf`;
      
      // Use direct HTTP download instead of blob URLs to avoid sandbox URL issues in Docker
      const downloadUrl = `${apiService.baseURL}/documents/download/${documentInfo.document_id}`;
      
      // Create download link with direct URL
      const link = document.createElement('a');
      link.href = downloadUrl;
      link.download = filename;
      link.style.display = 'none';
      link.target = '_blank'; // Open in new tab as fallback
      
      // Trigger download
      document.body.appendChild(link);
      link.click();
      
      // Show success message
      setDownloadSuccess('Download started successfully');
      
      // Clean up
      setTimeout(() => {
        try {
          document.body.removeChild(link);
        } catch (error) {
          console.warn('Cleanup error:', error);
        }
      }, 100);
      
      if (onDownloadComplete) {
        onDownloadComplete(documentInfo, filename);
      }
      
    } catch (error) {
      console.error('Download failed:', error);
      console.log('Document info that failed:', documentInfo);
      
      // Handle different types of download errors with specific messages
      let errorMessage = 'Download failed. Please try again.';
      
      if (error.message.includes('Document ID is missing')) {
        errorMessage = 'Document information is incomplete. Please generate the document again.';
      } else if (error.message.includes('not ready for download')) {
        errorMessage = 'Document generation is still in progress. Please wait a moment and try again.';
      } else if (error.message.includes('Invalid document ID format')) {
        errorMessage = 'Document ID is corrupted. Please generate a new document.';
      } else if (error.message.includes('temp-') || error.message.includes('generate')) {
        errorMessage = error.message; // Use the specific error message
      } else {
        errorMessage = 'Download failed. The document may have expired (24-hour limit) or been removed from the server.';
      }
      
      setDownloadError(errorMessage);
      
      if (onError) {
        onError(error, documentInfo);
      }
    } finally {
      setIsDownloading(false);
    }
  };

  // Retry download function
  const retryDownload = () => {
    setDownloadError(null);
    setDownloadSuccess(null);
    handleDownload();
  };
  
  const formatDocumentType = (type) => {
    return type?.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()) || 'Document';
  };
  
  const formatFileSize = (bytes) => {
    if (!bytes) return '';
    const sizes = ['B', 'KB', 'MB'];
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return Math.round(bytes / Math.pow(1024, i) * 100) / 100 + ' ' + sizes[i];
  };
  
  const isExpired = () => {
    if (!documentInfo?.expires_at) return false;
    return new Date(documentInfo.expires_at) < new Date();
  };
  
  if (!documentInfo) {
    return null;
  }
  
  if (isExpired()) {
    return (
      <div className="download-button-container expired">
        <div className="download-info">
          <span className="document-name">{formatDocumentType(documentInfo.document_type)}</span>
          <span className="download-status expired">Document Expired</span>
        </div>
        <button className="download-btn expired" disabled>
          <span className="download-icon">⚠️</span>
          Expired
        </button>
      </div>
    );
  }
  
  return (
    <div className="download-button-container">
      <div className="download-info">
        <div className="document-details">
          <span className="document-name">{formatDocumentType(documentInfo.document_type)}</span>
          <span className="employee-name">for {documentInfo.employee_name}</span>
        </div>
        <div className="document-meta">
          <span className="generated-time">
            Generated: {new Date(documentInfo.generated_at).toLocaleString()}
          </span>
          <span className="expiry-time">
            Expires: {new Date(documentInfo.expires_at).toLocaleString()}
          </span>
        </div>
      </div>
      
      <div className="download-actions">
        <button 
          className={`download-btn ${isDownloading ? 'downloading' : ''}`}
          onClick={handleDownload}
          disabled={isDownloading}
          title={`Download ${formatDocumentType(documentInfo.document_type)}`}
        >
          <span className="download-icon">
            {isDownloading ? '⏳' : '📄'}
          </span>
          <span className="download-text">
            {isDownloading ? 'Downloading...' : 'Download PDF'}
          </span>
        </button>
      </div>
      
      {downloadSuccess && (
        <div className="download-success">
          <span className="success-icon">✅</span>
          <span className="success-message">{downloadSuccess}</span>
          <button 
            className="dismiss-btn"
            onClick={() => setDownloadSuccess(null)}
            title="Dismiss"
          >
            ✕
          </button>
        </div>
      )}
      
      {downloadError && (
        <div className="download-error">
          <span className="error-icon">❌</span>
          <span className="error-message">{downloadError}</span>
          <button 
            className="retry-btn"
            onClick={retryDownload}
            disabled={isDownloading}
          >
            {isDownloading ? 'Retrying...' : 'Retry'}
          </button>
          <button 
            className="dismiss-btn"
            onClick={() => setDownloadError(null)}
            title="Dismiss"
          >
            ✕
          </button>
        </div>
      )}
    </div>
  );
};

export default DownloadButton;