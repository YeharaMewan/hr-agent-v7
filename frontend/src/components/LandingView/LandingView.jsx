import React, { useState, useRef, useEffect } from 'react';
import SuggestionChips from '../SuggestionChips/SuggestionChips';

const LandingView = ({ onFirstMessage }) => {
  const [inputValue, setInputValue] = useState('');
  const inputRef = useRef(null);
  const [isMobile, setIsMobile] = useState(false);
  const [isKeyboardOpen, setIsKeyboardOpen] = useState(false);
  const [isInputFocused, setIsInputFocused] = useState(false);
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

  // Enhanced mobile input focus handling
  const handleInputFocus = () => {
    setIsInputFocused(true);
    if (isMobile && inputRef.current) {
      // Prevent zoom on iOS
      inputRef.current?.setAttribute('readonly', 'readonly');
      inputRef.current?.removeAttribute('readonly');
    }
  };

  const handleInputBlur = () => {
    setIsInputFocused(false);
  };

  // Dynamic keyboard height calculation for precise alignment with keyboard edge
  useEffect(() => {
    if (isKeyboardOpen && isMobile) {
      const setKeyboardHeight = () => {
        const availableHeight = window.visualViewport?.height || window.innerHeight;
        const keyboardHeight = originalViewportHeight - availableHeight;
        // Use full keyboard height for precise bottom alignment - no multiplier
        document.documentElement.style.setProperty('--keyboard-height', `${Math.max(keyboardHeight, 0)}px`);
      };
      
      setKeyboardHeight();
      
      if (window.visualViewport) {
        window.visualViewport.addEventListener('resize', setKeyboardHeight);
        return () => {
          window.visualViewport.removeEventListener('resize', setKeyboardHeight);
        };
      }
    } else {
      document.documentElement.style.setProperty('--keyboard-height', '0px');
    }
  }, [isKeyboardOpen, isMobile, originalViewportHeight]);

  // Working quick actions - focused on queries and reports (attendance creation/updates not implemented)
  const suggestionChips = [
    { emoji: '📊', text: 'Attendance Overview', command: 'Show me attendance statistics and insights for the last 30 days' },
    { emoji: '📈', text: 'Department Trends', command: 'Give me attendance trends analysis for all departments' },
    { emoji: '👥', text: 'Team Performance', command: 'Show comparative attendance report for different departments' },
    { emoji: '🔍', text: 'Pattern Analysis', command: 'Provide attendance pattern analysis and recommendations' },
    { emoji: '👤', text: 'Employee Search', command: 'Find employee details and information' },
    { emoji: '📋', text: 'Generate Report', command: 'Create attendance summary report for last week' },
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
          onFocus={handleInputFocus}
          onBlur={handleInputBlur}
          onTouchStart={handleInputFocus}
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