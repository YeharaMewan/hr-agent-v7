import React from 'react';

const SuggestionChips = ({ chips, onChipClick, isLanding = false, isHidden = false }) => {
  const handleChipClick = (chip) => {
    onChipClick(chip.command);
  };

  const containerClass = isLanding 
    ? "suggestion-chips-container" 
    : `chat-suggestion-chips ${isHidden ? 'chips-hidden' : ''}`;

  return (
    <div className={containerClass}>
      {chips.map((chip, index) => (
        <button
          key={index}
          className="suggestion-chip"
          style={{ animationDelay: `${index * 100}ms` }}
          onClick={() => handleChipClick(chip)}
        >
          {chip.emoji} {chip.text}
        </button>
      ))}
    </div>
  );
};

export default SuggestionChips;