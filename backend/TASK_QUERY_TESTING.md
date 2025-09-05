# Intelligent Task Query System - Testing Guide

## Overview
This document provides comprehensive testing scenarios for the new intelligent task query system that was implemented. The system allows natural language queries about tasks with advanced filtering capabilities.

## Implementation Summary

### What Was Built
1. **Enhanced Data Models**: New Pydantic models for task queries and responses
2. **Intelligent Query Tool**: AI-powered natural language processing for task queries
3. **Dynamic SQL Builder**: Complex query construction across multiple tables
4. **LangGraph Integration**: Tool integrated with the AI agent system
5. **REST API Endpoints**: Both GET and POST endpoints for task querying

### Key Features Implemented
- **Time-based filtering**: "today", "yesterday", "last week", "last month"
- **Priority-based searches**: "urgent tasks", "high priority", "low priority"
- **People-based queries**: "John's tasks", "tasks led by Sarah"
- **Skill-based filtering**: "carpentry tasks", "electrical work"
- **Location-aware searches**: "office tasks", "remote work"
- **Smart response patterns**: Individual details, group summaries, statistics

## Testing Scenarios

### 1. Time-Based Query Tests

#### Test Cases:
```bash
# GET endpoint tests
curl -X GET "http://localhost:8000/tasks/intelligent?q=today's tasks"
curl -X GET "http://localhost:8000/tasks/intelligent?q=yesterday's tasks"
curl -X GET "http://localhost:8000/tasks/intelligent?q=this week tasks"
curl -X GET "http://localhost:8000/tasks/intelligent?q=last month tasks"

# POST endpoint tests
curl -X POST "http://localhost:8000/tasks/intelligent" \
  -H "Content-Type: application/json" \
  -d '{"query": "show me tasks from last week"}'
```

#### Expected Responses:
- **Individual tasks**: Full task details with leaders, labourers, skills
- **Multiple tasks**: List format with basic details
- **Large datasets**: Statistical summaries with breakdowns

### 2. Priority-Based Query Tests

#### Test Cases:
```bash
curl -X GET "http://localhost:8000/tasks/intelligent?q=urgent tasks"
curl -X GET "http://localhost:8000/tasks/intelligent?q=high priority tasks"
curl -X GET "http://localhost:8000/tasks/intelligent?q=low priority work"

# POST with complex priority queries
curl -X POST "http://localhost:8000/tasks/intelligent" \
  -H "Content-Type: application/json" \
  -d '{"query": "show me all urgent and high priority tasks from this month"}'
```

#### Expected Behavior:
- Filters tasks by priority level
- Combines with time filters when specified
- Provides priority breakdown in statistics

### 3. People-Based Query Tests

#### Test Cases:
```bash
curl -X GET "http://localhost:8000/tasks/intelligent?q=tasks led by John"
curl -X GET "http://localhost:8000/tasks/intelligent?q=Sarah's tasks"
curl -X GET "http://localhost:8000/tasks/intelligent?q=show me Mike Johnson tasks"

# POST with department leader queries
curl -X POST "http://localhost:8000/tasks/intelligent" \
  -H "Content-Type: application/json" \
  -d '{"query": "tasks managed by AI department leaders"}'
```

#### Expected Behavior:
- Searches through task group leaders
- Matches names with ILIKE pattern matching
- Shows leader information in response

### 4. Skill-Based Query Tests

#### Test Cases:
```bash
curl -X GET "http://localhost:8000/tasks/intelligent?q=carpentry tasks"
curl -X GET "http://localhost:8000/tasks/intelligent?q=electrical work"
curl -X GET "http://localhost:8000/tasks/intelligent?q=construction tasks"

# Combined skill and time queries
curl -X POST "http://localhost:8000/tasks/intelligent" \
  -H "Content-Type: application/json" \
  -d '{"query": "plumbing work from last week"}'
```

#### Expected Behavior:
- Filters through labour skills
- Shows labour names and skills in results
- Combines with other filters

### 5. Location-Based Query Tests

#### Test Cases:
```bash
curl -X GET "http://localhost:8000/tasks/intelligent?q=office tasks"
curl -X GET "http://localhost:8000/tasks/intelligent?q=remote work"
curl -X GET "http://localhost:8000/tasks/intelligent?q=downtown site tasks"

# Complex location queries
curl -X POST "http://localhost:8000/tasks/intelligent" \
  -H "Content-Type: application/json" \
  -d '{"query": "urgent tasks at the warehouse this week"}'
```

#### Expected Behavior:
- Searches task locations with ILIKE matching
- Provides location breakdown in statistics
- Combines with priority and time filters

### 6. Complex Multi-Filter Tests

#### Test Cases:
```bash
# Multiple criteria combinations
curl -X POST "http://localhost:8000/tasks/intelligent" \
  -H "Content-Type: application/json" \
  -d '{"query": "urgent carpentry tasks led by John this week"}'

curl -X POST "http://localhost:8000/tasks/intelligent" \
  -H "Content-Type: application/json" \
  -d '{"query": "high priority office work from last month"}'

curl -X POST "http://localhost:8000/tasks/intelligent" \
  -H "Content-Type: application/json" \
  -d '{"query": "show me all electrical tasks at downtown sites"}'
```

#### Expected Behavior:
- Combines multiple WHERE conditions with AND
- Shows applied filters in response
- Provides relevant follow-up questions

### 7. Edge Case Tests

#### Test Cases:
```bash
# No results
curl -X GET "http://localhost:8000/tasks/intelligent?q=nonexistent tasks"

# Very broad queries
curl -X GET "http://localhost:8000/tasks/intelligent?q=all tasks"

# Ambiguous queries
curl -X GET "http://localhost:8000/tasks/intelligent?q=tasks"
```

#### Expected Behavior:
- **No results**: Provides helpful suggestions and alternative queries
- **Broad queries**: Returns statistical summaries for large datasets
- **Ambiguous**: Asks clarifying questions through follow-up suggestions

## Response Structure Validation

### Individual Task Response
```json
{
  "success": true,
  "message": "Found 1 task: **Task Title**",
  "data": {
    "task_id": 1,
    "title": "Task Title",
    "priority": "High",
    "location": "Office",
    "expected_days": 5,
    "notes": "Task notes",
    "group_name": "Group Name",
    "leaders": "John Doe",
    "departments": "Engineering",
    "labourers": "Worker Name",
    "labour_skills": "Skill Name",
    "created_at": "2024-01-01 10:00"
  },
  "response_type": "individual_task",
  "total_found": 1,
  "follow_up_questions": ["..."],
  "filters_applied": {"priority": ["High"], "leaders": ["John"]}
}
```

### Multiple Tasks Response
```json
{
  "success": true,
  "message": "Found 5 tasks matching your criteria:",
  "data": [
    {
      "id": 1,
      "title": "Task 1",
      "priority": "High",
      "group": "Group A",
      "leaders": "John Doe",
      "location": "Office",
      "created_at": "2024-01-01"
    }
  ],
  "response_type": "multiple_tasks",
  "total_found": 5,
  "follow_up_questions": ["..."],
  "filters_applied": {"..."}
}
```

### Statistics Response
```json
{
  "success": true,
  "message": "Found 25 tasks. Here's a summary breakdown:",
  "data": {
    "total_tasks": 25,
    "priority_breakdown": {"High": 5, "Medium": 15, "Low": 5},
    "group_breakdown": {"Group A": 10, "Group B": 15},
    "location_breakdown": {"Office": 20, "Remote": 5},
    "sample_tasks": ["..."]
  },
  "response_type": "task_statistics",
  "total_found": 25,
  "follow_up_questions": ["..."]
}
```

## Chat Integration Testing

### Test through Chat Endpoint
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "show me today'\''s urgent tasks"}'

curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "what carpentry work do we have this week?"}'
```

### Expected Chat Behavior:
- AI agent automatically calls `query_tasks_intelligent()` tool
- Provides conversational responses based on tool results
- Offers follow-up questions and refinements
- Maintains context across conversation

## Performance Considerations

### Query Optimization
- Uses proper JOIN strategy across related tables
- Applies filters efficiently with parameterized queries
- Groups results to avoid duplicates from many-to-many relationships

### Response Optimization
- Adaptive response patterns based on result count
- Limits sample data for large result sets
- Provides statistical summaries instead of full data dumps

## Human-Loop Features

### Clarification Scenarios
1. **Ambiguous names**: "Did you mean John Smith or John Doe?"
2. **Too many results**: "Found 100+ tasks. Would you like to filter by priority?"
3. **No results**: "No tasks found. Try searching for a different time period?"
4. **Alternative suggestions**: "Show tasks from last week only?"

### Follow-up Questions
- Automatically generated based on query context
- Suggest logical next steps (filter by priority, show details, etc.)
- Provide alternative query patterns

## Integration Points

### Database Tables Used
- `tasks` (main task data)
- `task_groups` (task organization)
- `task_group_leaders` (leader assignments)  
- `task_labours` (labour assignments)
- `labours` (worker skills)
- `employees` (leader information)
- `departments` (organizational data)

### LangGraph Integration
- Tool registered in main tools list
- System prompt includes task query instructions
- Automatic tool invocation for task-related queries

## Success Metrics

### Functional Tests
- ✅ All query patterns work correctly
- ✅ SQL queries execute without errors
- ✅ Response formats match specifications
- ✅ Edge cases handled gracefully

### Performance Tests
- ✅ Response times under acceptable limits
- ✅ Database queries optimized
- ✅ Memory usage reasonable for large result sets

### User Experience Tests
- ✅ Natural language understanding works
- ✅ Follow-up questions are relevant
- ✅ Error messages are helpful
- ✅ Chat integration seamless

## Conclusion

The intelligent task query system provides a comprehensive, AI-powered solution for querying tasks using natural language. It successfully integrates with the existing HR management system and provides sophisticated filtering, intelligent responses, and human-loop guidance.

The system handles various query patterns, from simple time-based searches to complex multi-criteria filters, while maintaining performance and providing excellent user experience through conversational AI interactions.