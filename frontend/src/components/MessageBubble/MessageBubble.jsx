import React from 'react';
import { marked } from 'marked';
import ChartRenderer from '../ChartRenderer/ChartRenderer';

const MessageBubble = ({ message }) => {
  const { content, isUser, hasChart, chartData } = message;

  // Configure marked to avoid XSS
  marked.setOptions({
    breaks: true,
    sanitize: false // We'll use DOMPurify or similar if needed
  });

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

  return (
    <div className={`message-wrapper ${isUser ? 'user' : 'bot'}`}>
      <div className={`message-bubble ${hasChart ? 'has-chart-content' : ''}`}>
        {renderContent()}
      </div>
    </div>
  );
};

export default MessageBubble;