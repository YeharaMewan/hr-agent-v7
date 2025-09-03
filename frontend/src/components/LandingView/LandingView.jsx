import React, { useState, useRef, useEffect } from 'react';
import SuggestionChips from '../SuggestionChips/SuggestionChips';

const LandingView = ({ onFirstMessage }) => {
  const [inputValue, setInputValue] = useState('');
  const inputRef = useRef(null);
  const [isMobile, setIsMobile] = useState(false);
  const [isKeyboardOpen, setIsKeyboardOpen] = useState(false);
  const [originalViewportHeight, setOriginalViewportHeight] = useState(window.innerHeight);

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

  // Keyboard visibility detection for mobile
  useEffect(() => {
    if (!isMobile) return;

    const handleViewportChange = () => {
      const currentHeight = window.visualViewport?.height || window.innerHeight;
      const heightDifference = originalViewportHeight - currentHeight;
      const keyboardThreshold = 150;
      
      const keyboardIsOpen = heightDifference > keyboardThreshold;
      setIsKeyboardOpen(keyboardIsOpen);
    };

    if (window.visualViewport) {
      window.visualViewport.addEventListener('resize', handleViewportChange);
      return () => {
        window.visualViewport.removeEventListener('resize', handleViewportChange);
      };
    } else {
      window.addEventListener('resize', handleViewportChange);
      return () => {
        window.removeEventListener('resize', handleViewportChange);
      };
    }
  }, [isMobile, originalViewportHeight]);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (inputValue.trim()) {
      // Hide keyboard on mobile before sending message
      if (isMobile && inputRef.current) {
        inputRef.current.blur();
      }
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
    // Hide keyboard on mobile before sending message
    if (isMobile && inputRef.current) {
      inputRef.current.blur();
    }
    onFirstMessage(command);
  };

  // Handle mobile input focus
  const handleMobileInputFocus = () => {
    if (isMobile && inputRef.current) {
      setTimeout(() => {
        inputRef.current?.focus();
        // Prevent zoom on iOS
        inputRef.current?.setAttribute('readonly', 'readonly');
        inputRef.current?.removeAttribute('readonly');
      }, 100);
    }
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
    <div className={`landing-container ${isMobile ? 'mobile' : ''} ${isKeyboardOpen ? 'keyboard-open' : ''}`}>
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
          ref={inputRef}
          type="text"
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyPress={handleKeyPress}
          onFocus={handleMobileInputFocus}
          onTouchStart={handleMobileInputFocus}
          className="landing-input"
          placeholder="Ask anything about HR..."
          autoComplete="off"
          inputMode={isMobile ? "text" : undefined}
        />
      </form>
    </div>
  );
};

export default LandingView;