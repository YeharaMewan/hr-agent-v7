import React, { useState } from 'react';
import SuggestionChips from '../SuggestionChips/SuggestionChips';

const LandingView = ({ onFirstMessage }) => {
  const [inputValue, setInputValue] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (inputValue.trim()) {
      onFirstMessage(inputValue);
      setInputValue('');
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const handleChipClick = (command) => {
    onFirstMessage(command);
  };

  // Enhanced attendance-focused landing chips
  const suggestionChips = [
    { emoji: '✅', text: 'Mark Attendance', command: 'Mark my attendance as Present for today' },
    { emoji: '📊', text: 'Attendance Overview', command: 'Show me attendance statistics and insights for the last 30 days' },
    { emoji: '🏠', text: 'Work From Home', command: 'Mark my attendance as Work from home for today' },
    { emoji: '📈', text: 'Department Trends', command: 'Give me attendance trends analysis for all departments' },
    { emoji: '👥', text: 'Team Performance', command: 'Show comparative attendance report for different departments' },
    { emoji: '🔍', text: 'Pattern Analysis', command: 'Provide attendance pattern analysis and recommendations' },
  ];

  return (
    <div className="landing-container">
      <div className="title-block">
        <img src="/RiseHRLogo.png" alt="RiseHR Logo" className="logo" />
        <p className="subtitle">Your Intelligent HR Partner</p>
      </div>

      <SuggestionChips 
        chips={suggestionChips} 
        onChipClick={handleChipClick}
        isLanding={true}
      />

      <form onSubmit={handleSubmit} className="new-input-area">
        <input
          type="text"
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyPress={handleKeyPress}
          className="landing-input"
          placeholder="Ask anything about HR..."
          autoComplete="off"
        />
      </form>
    </div>
  );
};

export default LandingView;