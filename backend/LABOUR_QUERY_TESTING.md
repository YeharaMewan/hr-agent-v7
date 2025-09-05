# Intelligent Labour Query System - Testing Guide

## Overview
This document provides comprehensive testing scenarios for the new intelligent labour query system. The system allows natural language queries about labourers, their skills, workload analysis, and task assignments with advanced filtering capabilities.

## Implementation Summary

### What Was Built
1. **Enhanced Labour Data Models**: New Pydantic models for labour queries and responses with workload analysis
2. **Intelligent Labour Query Tool**: AI-powered natural language processing for labour-specific queries
3. **Dynamic Workload Analysis SQL**: Complex query construction with real-time workload status calculation
4. **LangGraph Integration**: Tool integrated with the AI agent system for conversational queries
5. **REST API Endpoints**: Multiple endpoints for labour querying (intelligent, basic, filtered)

### Key Features Implemented
- **Skill-based filtering**: "carpenters", "electricians", "plumbers", "painters"
- **Workload analysis**: "available workers", "busy labourers", "overloaded staff"
- **Individual profiles**: "John the carpenter", "Sarah's workload"
- **Task assignment tracking**: "who worked on urgent tasks", "labourers on projects"
- **Time-based filtering**: "this week's workers", "last month's labour"
- **Smart response patterns**: Individual details, skill groups, workload analysis, statistics
- **Actionable recommendations**: workload balancing, skill optimization, task redistribution

## Testing Scenarios

### 1. Skill-Based Query Tests

#### Test Cases:
```bash
# GET endpoint tests
curl -X GET "http://localhost:8000/labourers/intelligent?q=show me all carpenters"
curl -X GET "http://localhost:8000/labourers/intelligent?q=available electricians"
curl -X GET "http://localhost:8000/labourers/intelligent?q=plumbing workers"
curl -X GET "http://localhost:8000/labourers/intelligent?q=construction labourers"

# POST endpoint tests
curl -X POST "http://localhost:8000/labourers/intelligent" \
  -H "Content-Type: application/json" \
  -d '{"query": "show me all electrical workers with their current workload"}'
```

#### Expected Responses:
- **Single skill group**: Organized list of workers with same skill and workload distribution
- **Mixed skills**: Statistical breakdown with skill categories
- **Workload insights**: Available vs busy workers within skill categories

### 2. Workload Analysis Query Tests

#### Test Cases:
```bash
curl -X GET "http://localhost:8000/labourers/intelligent?q=available workers"
curl -X GET "http://localhost:8000/labourers/intelligent?q=busy labourers"
curl -X GET "http://localhost:8000/labourers/intelligent?q=overloaded staff"
curl -X GET "http://localhost:8000/labourers/intelligent?q=workload analysis"

# Complex workload queries
curl -X POST "http://localhost:8000/labourers/intelligent" \
  -H "Content-Type: application/json" \
  -d '{"query": "show me overloaded carpenters who need help"}'
```

#### Expected Behavior:
- **Available workers**: Shows labourers with 0 current tasks
- **Busy workers**: Shows workers with moderate to heavy workloads
- **Overloaded analysis**: Identifies workers with too many tasks + recommendations
- **Workload statistics**: Comprehensive breakdown with actionable insights

### 3. Individual Labour Profile Tests

#### Test Cases:
```bash
curl -X GET "http://localhost:8000/labourers/intelligent?q=John the carpenter"
curl -X GET "http://localhost:8000/labourers/intelligent?q=show me Sarah details"
curl -X GET "http://localhost:8000/labourers/intelligent?q=Mike's current workload"

# POST with specific worker queries
curl -X POST "http://localhost:8000/labourers/intelligent" \
  -H "Content-Type: application/json" \
  -d '{"query": "give me complete profile of electrician David"}'
```

#### Expected Behavior:
- Full individual profile with current task assignments
- Workload analysis with recommendations
- Working relationships (leaders, departments)
- Task history and skill utilization

### 4. Task Assignment Query Tests

#### Test Cases:
```bash
curl -X GET "http://localhost:8000/labourers/intelligent?q=who worked on urgent tasks"
curl -X GET "http://localhost:8000/labourers/intelligent?q=labourers on downtown projects"
curl -X GET "http://localhost:8000/labourers/intelligent?q=workers assigned to high priority"

# Combined task and skill queries
curl -X POST "http://localhost:8000/labourers/intelligent" \
  -H "Content-Type: application/json" \
  -d '{"query": "carpenters working on urgent tasks this week"}'
```

#### Expected Behavior:
- Filters labourers by task priority/type
- Shows task assignment relationships
- Cross-references skills with task requirements
- Provides task assignment patterns

### 5. Time-Based Labour Query Tests

#### Test Cases:
```bash
curl -X GET "http://localhost:8000/labourers/intelligent?q=this week's workers"
curl -X GET "http://localhost:8000/labourers/intelligent?q=labourers from last month"
curl -X GET "http://localhost:8000/labourers/intelligent?q=today's labour assignments"

# Complex time-based queries
curl -X POST "http://localhost:8000/labourers/intelligent" \
  -H "Content-Type: application/json" \
  -d '{"query": "electricians who worked on projects last week"}'
```

#### Expected Behavior:
- Time-filtered labour assignments
- Historical workload patterns
- Skill utilization over time periods
- Task completion tracking

### 6. Complex Multi-Criteria Tests

#### Test Cases:
```bash
# Multiple criteria combinations
curl -X POST "http://localhost:8000/labourers/intelligent" \
  -H "Content-Type: application/json" \
  -d '{"query": "available carpenters for urgent task assignment"}'

curl -X POST "http://localhost:8000/labourers/intelligent" \
  -H "Content-Type: application/json" \
  -d '{"query": "overloaded electrical workers who need task redistribution"}'

curl -X POST "http://localhost:8000/labourers/intelligent" \
  -H "Content-Type: application/json" \
  -d '{"query": "show me skill distribution and workload balance analysis"}'
```

#### Expected Behavior:
- Combines skill + workload + availability filters
- Provides actionable recommendations
- Shows cross-skill workload balancing opportunities
- Suggests task redistribution strategies

### 7. Basic Labour Endpoint Tests

#### Test Cases:
```bash
# Basic filtering without AI
curl -X GET "http://localhost:8000/labourers?skill=carpenter"
curl -X GET "http://localhost:8000/labourers?available_only=true"
curl -X GET "http://localhost:8000/labourers?name=John&skill=electrician"

# Combined filters
curl -X GET "http://localhost:8000/labourers?skill=plumber&available_only=true"
```

#### Expected Behavior:
- Standard REST API filtering
- Workload status calculation
- Simple availability filtering
- Name and skill search

### 8. Edge Case Tests

#### Test Cases:
```bash
# No results
curl -X GET "http://localhost:8000/labourers/intelligent?q=nonexistent skill workers"

# Very broad queries
curl -X GET "http://localhost:8000/labourers/intelligent?q=all workers"

# Ambiguous queries
curl -X GET "http://localhost:8000/labourers/intelligent?q=workers"

# Conflicting filters
curl -X GET "http://localhost:8000/labourers/intelligent?q=available overloaded workers"
```

#### Expected Behavior:
- **No results**: Helpful suggestions with similar skills and alternative queries
- **Broad queries**: Statistical summaries with skill/workload breakdowns
- **Ambiguous**: Clarifying follow-up questions
- **Conflicting**: Smart resolution with explanation

## Response Structure Validation

### Individual Labour Response
```json
{
  "success": true,
  "message": "Found labourer: **John Smith** (Carpenter)",
  "data": {
    "labour_id": 1,
    "name": "John Smith",
    "skill": "Carpenter",
    "workload_status": "moderate_load",
    "current_task_count": 3,
    "current_tasks": "Kitchen Renovation, Deck Building, Cabinet Installation",
    "task_groups": "Residential, Commercial",
    "working_under_leaders": "Sarah Johnson, Mike Davis",
    "leader_departments": "Construction, Maintenance",
    "average_task_priority": 2.3,
    "created_at": "2024-01-15"
  },
  "response_type": "individual_labour",
  "total_found": 1,
  "workload_analysis": {
    "status": "moderate_load",
    "task_count": 3,
    "average_priority": 2.3,
    "recommendation": "Well-balanced workload"
  },
  "follow_up_questions": [
    "Show other carpenters with similar workload?",
    "Find available workers for task assignment?",
    "Show workload analysis for all workers?"
  ]
}
```

### Skill Group Response
```json
{
  "success": true,
  "message": "Found 5 carpenter workers:",
  "data": [
    {
      "id": 1,
      "name": "John Smith",
      "skill": "Carpenter",
      "workload_status": "moderate_load",
      "task_count": 3,
      "current_tasks": "Kitchen Renovation, Deck Building",
      "leaders": "Sarah Johnson"
    }
  ],
  "response_type": "skill_group",
  "skill_name": "carpenter",
  "total_found": 5,
  "workload_distribution": {
    "available": 1,
    "light_load": 1,
    "moderate_load": 2,
    "heavy_load": 1,
    "overloaded": 0
  },
  "follow_up_questions": [
    "Show only available carpenters?",
    "Find busy carpenters?",
    "Get detailed workload analysis?"
  ]
}
```

### Labour Statistics Response
```json
{
  "success": true,
  "message": "Found 25 labourers. Here's a comprehensive analysis:",
  "data": {
    "total_labourers": 25,
    "skill_breakdown": {
      "Carpenter": 8,
      "Electrician": 6,
      "Plumber": 4,
      "Painter": 4,
      "Mason": 3
    },
    "workload_breakdown": {
      "available": 5,
      "light_load": 7,
      "moderate_load": 8,
      "heavy_load": 4,
      "overloaded": 1
    },
    "task_assignment_patterns": {
      "No tasks": 5,
      "1-2 tasks": 7,
      "3-4 tasks": 8,
      "5+ tasks": 5
    },
    "sample_labourers": ["..."]
  },
  "response_type": "labour_statistics",
  "total_labourers": 25,
  "recommendations": [
    "Consider redistributing tasks from 1 overloaded worker to 5 available workers",
    "You have 5 available workers ready for new assignments"
  ],
  "follow_up_questions": [
    "Show only available workers?",
    "Filter by specific skill (carpenter, electrician, etc.)?",
    "Find overloaded workers who need help?"
  ]
}
```

## Workload Status Calculation

### Workload Categories
- **available**: 0 current tasks
- **light_load**: 1-2 current tasks  
- **moderate_load**: 3-4 current tasks
- **heavy_load**: 5-6 current tasks
- **overloaded**: 7+ current tasks

### SQL Implementation
```sql
CASE 
    WHEN COUNT(DISTINCT tl.task_id) = 0 THEN 'available'
    WHEN COUNT(DISTINCT tl.task_id) <= 2 THEN 'light_load'
    WHEN COUNT(DISTINCT tl.task_id) <= 4 THEN 'moderate_load'
    WHEN COUNT(DISTINCT tl.task_id) <= 6 THEN 'heavy_load'
    ELSE 'overloaded'
END as workload_status
```

## Chat Integration Testing

### Test through Chat Endpoint
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "show me all available carpenters"}'

curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "who are the overloaded workers that need help?"}'

curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "give me John the carpenter details"}'
```

### Expected Chat Behavior:
- AI agent automatically calls `query_labourers_intelligent()` tool
- Provides conversational responses with workload recommendations
- Offers actionable follow-up questions
- Maintains context for labour management discussions

## Human-Loop Features

### Intelligent Recommendations
1. **Workload Balancing**: "Consider redistributing tasks from overloaded workers"
2. **Skill Optimization**: "High concentration of carpenters - consider cross-training"  
3. **Availability Insights**: "You have 5 available workers ready for assignments"
4. **Task Distribution**: "These workers can take on additional tasks"

### Follow-up Questions
- Skill-specific: "Show only available carpenters?"
- Workload-focused: "Find overloaded workers who need help?"
- Analysis-driven: "Get detailed workload analysis?"
- Action-oriented: "See task redistribution opportunities?"

## Performance Considerations

### Query Optimization
- Efficient JOINs across labour, task, and assignment tables
- Workload calculation performed in SQL for better performance
- Parameterized queries for security and caching
- GROUP BY optimization for aggregated workload data

### Response Optimization
- Adaptive response patterns based on result count and query type
- Statistical summaries for large datasets
- Workload analysis integrated into responses
- Actionable recommendations generated dynamically

## Integration Points

### Database Relationships Used
- `labours` (main labour data)
- `task_labours` (task assignments)
- `tasks` (task details)
- `task_groups` (task organization)
- `task_group_leaders` (leadership relationships)
- `employees` (leader information)
- `departments` (organizational structure)

### LangGraph Integration
- Tool registered in main tools list as `query_labourers_intelligent`
- System prompt includes labour query instructions
- Automatic tool invocation for labour-related queries
- Human-loop recommendations integrated into conversations

## Success Metrics

### Functional Tests
- ✅ All skill-based queries work correctly
- ✅ Workload analysis functions accurately  
- ✅ Individual profiles show complete information
- ✅ Task assignment queries filter properly
- ✅ Time-based filtering works as expected
- ✅ Complex multi-criteria queries handled
- ✅ Edge cases managed gracefully

### Performance Tests
- ✅ Response times under acceptable limits
- ✅ SQL queries optimized for workload calculation
- ✅ Memory usage reasonable for large labour datasets
- ✅ Real-time workload status calculation efficient

### User Experience Tests
- ✅ Natural language understanding for labour queries
- ✅ Workload recommendations are actionable
- ✅ Follow-up questions relevant to labour management
- ✅ Error messages provide helpful alternatives
- ✅ Chat integration seamless for labour discussions

## Conclusion

The intelligent labour query system provides comprehensive, AI-powered labour management with sophisticated workload analysis, skill-based filtering, and actionable recommendations. It successfully integrates with the existing HR system while providing unique labour-focused insights.

The system handles various query patterns from simple skill searches to complex workload analysis, maintaining excellent performance while providing conversational AI interactions that help optimize labour allocation and task distribution.

## Key Differentiators from Task System

1. **Workload Analysis**: Real-time calculation of labour availability and task load
2. **Skill Optimization**: Recommendations for skill distribution and cross-training
3. **Task Assignment Tracking**: Shows which labourers work on what types of tasks
4. **Availability Management**: Identifies available workers for new assignments
5. **Redistribution Recommendations**: Suggests task rebalancing for overloaded workers
6. **Individual Profiles**: Complete worker profiles with performance insights
7. **Labour Analytics**: Comprehensive workforce analysis with actionable intelligence