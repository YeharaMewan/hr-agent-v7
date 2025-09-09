import React from 'react';
import { marked } from 'marked';
import ChartRenderer from '../ChartRenderer/ChartRenderer';
import DownloadButton from '../DownloadButton/DownloadButton';
import './MessageBubble.css';

const MessageBubble = ({ message }) => {
  const { content, isUser, hasChart, chartData, documentInfo } = message;

  // Configure marked to avoid XSS
  marked.setOptions({
    breaks: true,
    sanitize: false // We'll use DOMPurify or similar if needed
  });

  // Parse document information from message content if not explicitly provided
  const parseDocumentInfo = () => {
    console.log('MessageBubble parseDocumentInfo - documentInfo prop:', documentInfo);
    console.log('MessageBubble parseDocumentInfo - isUser:', isUser);
    console.log('MessageBubble parseDocumentInfo - content preview:', content?.substring(0, 100));
    
    if (documentInfo) {
      console.log('MessageBubble: Using provided documentInfo prop:', documentInfo);
      
      // Validate that documentInfo has a proper UUID document_id (not a temp ID)
      if (documentInfo.document_id && !documentInfo.document_id.startsWith('temp-')) {
        const uuidPattern = /^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$/i;
        if (uuidPattern.test(documentInfo.document_id)) {
          console.log('MessageBubble: Valid UUID document ID found:', documentInfo.document_id);
          return documentInfo;
        } else {
          console.warn('MessageBubble: Invalid document_id format in documentInfo:', documentInfo.document_id);
        }
      } else {
        console.warn('MessageBubble: documentInfo has temp or missing document_id:', documentInfo.document_id);
      }
    }
    if (isUser) return null;
    
    try {
      // First try to parse as complete JSON response from AI
      const jsonObjectPattern = /"document_info":\s*\{[^}]+\}/g;
      const jsonMatch = content.match(jsonObjectPattern);
      
      if (jsonMatch) {
        try {
          // Extract the document_info object
          const docInfoString = jsonMatch[0].replace('"document_info":', '');
          const parsed = JSON.parse(docInfoString);
          console.log('MessageBubble: Successfully parsed document_info:', parsed);
          return parsed;
        } catch (parseError) {
          console.warn('Failed to parse document_info JSON:', parseError);
        }
      }
      
      // Look for complete JSON response in content
      const fullJsonPattern = /\{[^{}]*"success":\s*true[^{}]*"document_info":[^{}]*\{[^}]+\}[^{}]*\}/g;
      const fullJsonMatch = content.match(fullJsonPattern);
      
      if (fullJsonMatch) {
        try {
          const parsed = JSON.parse(fullJsonMatch[0]);
          if (parsed.document_info) {
            console.log('MessageBubble: Successfully parsed full JSON with document_info:', parsed.document_info);
            return parsed.document_info;
          }
        } catch (parseError) {
          console.warn('Failed to parse full JSON response:', parseError);
        }
      }
      
      // Look for document information in the message content (fallback)
      const documentPattern = /document_info.*?({[^}]+})/gi;
      const match = content.match(documentPattern);
      
      if (match) {
        // Try to extract JSON-like structure from content
        const jsonMatch = content.match(/{\s*"success":\s*true[^}]+}/g);
        if (jsonMatch) {
          const parsed = JSON.parse(jsonMatch[0]);
          return parsed.document_info || null;
        }
      }
      
      // Enhanced: Look for document generation success indicators
      if (content.includes('generated successfully') || 
          (content.includes('✅') && content.includes('download'))) {
        
        // Try multiple patterns for document ID extraction
        const documentIdPatterns = [
          /document[_-]id['":\s]*([a-f0-9-]{36})/i,
          /id['":\s]*([a-f0-9-]{36})/i,
          /([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})/i
        ];
        
        let documentIdMatch = null;
        for (const pattern of documentIdPatterns) {
          documentIdMatch = content.match(pattern);
          if (documentIdMatch) break;
        }
        
        // Try multiple patterns for employee name - more restrictive
        const employeeNamePatterns = [
          /generated successfully for\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*?)(?:\s|$|\.)/i,
          /(?:for|employee['":\s]*)['":\s]*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*?)(?:\s|$|[.,])/i,
          /name['":\s]*['":]([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*?)['",]/i
        ];
        
        let employeeNameMatch = null;
        for (const pattern of employeeNamePatterns) {
          employeeNameMatch = content.match(pattern);
          if (employeeNameMatch) break;
        }
        
        // Enhanced document type detection
        const documentTypePatterns = [
          { pattern: /confirmation\s+letter/i, type: 'confirmation_letter' },
          { pattern: /service\s+letter/i, type: 'service_letter' },
          { pattern: /offer\s+letter/i, type: 'offer_letter' },
          { pattern: /reference\s+letter/i, type: 'reference_letter' }
        ];
        
        let documentType = 'service_letter'; // default
        for (const { pattern, type } of documentTypePatterns) {
          if (pattern.test(content)) {
            documentType = type;
            break;
          }
        }
        
        const filenameMatch = content.match(/filename['":\s]*['":]([^,\s'"]+)/i);
        
        // If we found either document ID or employee name, create document info
        // BUT only create temp IDs if we don't have a proper document ID
        if (documentIdMatch || employeeNameMatch) {
          // Don't create temp IDs - only create document info if we have a real UUID
          if (!documentIdMatch) {
            console.warn('MessageBubble: Document generation detected but no valid document ID found. Skipping temp ID creation to prevent download errors.');
            return null;
          }
          
          const docInfo = {
            document_id: documentIdMatch[1],
            employee_name: employeeNameMatch ? employeeNameMatch[1] : 'Unknown',
            document_type: documentType,
            filename: filenameMatch ? filenameMatch[1] : null,
            generated_at: new Date().toISOString(),
            expires_at: new Date(Date.now() + 24 * 60 * 60 * 1000).toISOString()
          };
          
          console.log('MessageBubble: Parsed document info from content with valid UUID:', docInfo);
          return docInfo;
        } else {
          console.warn('MessageBubble: Document generation detected but could not extract details from:', content);
        }
      }
      
      return null;
    } catch (error) {
      console.warn('Failed to parse document info:', error);
      return null;
    }
  };

  const extractedDocumentInfo = parseDocumentInfo();

  const handleDownloadStart = (docInfo) => {
    console.log('Download started for:', docInfo);
  };

  const handleDownloadComplete = (docInfo, filename) => {
    console.log('Download completed:', filename);
    
    // Could add a success notification here
    const notification = document.createElement('div');
    notification.textContent = `✅ ${filename} downloaded successfully!`;
    notification.style.cssText = `
      position: fixed;
      top: 20px;
      right: 20px;
      background: #4caf50;
      color: white;
      padding: 12px 20px;
      border-radius: 8px;
      z-index: 1000;
      box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    `;
    document.body.appendChild(notification);
    
    setTimeout(() => {
      document.body.removeChild(notification);
    }, 3000);
  };

  const handleDownloadError = (error, docInfo) => {
    console.error('Download failed:', error, docInfo);
  };

  const renderContent = () => {
    if (hasChart && chartData) {
      return <ChartRenderer chartData={chartData} />;
    }

    if (isUser) {
      // For user messages, escape HTML to prevent XSS
      return content.replace(/</g, "&lt;").replace(/>/g, "&gt;");
    } else {
      // For bot messages, render markdown
      return <div dangerouslySetInnerHTML={{ __html: marked(content) }} />;
    }
  };

  const hasDocumentContent = extractedDocumentInfo || documentInfo;

  return (
    <div className={`message-wrapper ${isUser ? 'user' : 'bot'}`}>
      <div className={`message-bubble ${hasChart ? 'has-chart-content' : ''} ${hasDocumentContent ? 'has-document-content' : ''}`}>
        {renderContent()}
        
        {hasDocumentContent && !isUser && (
          <div className="document-section">
            <DownloadButton
              documentInfo={extractedDocumentInfo || documentInfo}
              onDownloadStart={handleDownloadStart}
              onDownloadComplete={handleDownloadComplete}
              onError={handleDownloadError}
            />
          </div>
        )}
      </div>
    </div>
  );
};

export default MessageBubble;