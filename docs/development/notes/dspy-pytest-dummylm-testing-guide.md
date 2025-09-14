# DSPy Unit Testing with DummyLM: Complete Implementation Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Setup and Configuration](#setup-and-configuration)
3. [Understanding DummyLM Modes](#understanding-dummylm-modes)
4. [Working Examples](#working-examples)
5. [Best Practices](#best-practices)
6. [Troubleshooting](#troubleshooting)

## Introduction

DummyLM is a deterministic, cost-free mock language model designed specifically for unit testing DSPy applications. It enables fast, reliable testing without API calls, making it ideal for CI/CD pipelines and rapid development cycles.

### Key Benefits
- **Zero cost**: No API calls or tokens consumed
- **Deterministic**: Predictable outputs for reliable tests
- **Fast execution**: No network latency
- **Three operational modes**: List, Dictionary, and Example-following
- **Full async support**: Test async workflows seamlessly

## Setup and Configuration

### Installation Requirements
```python
import pytest
import dspy
from dspy.utils import DummyLM
```

### Basic Configuration Patterns

#### Method 1: Global Configuration
```python
def test_basic_dspy_module():
    lm = DummyLM([{"answer": "test response"}])
    dspy.settings.configure(lm=lm)

    predictor = dspy.Predict("question -> answer")
    result = predictor(question="What's the weather?")
    assert result.answer == "test response"
```

#### Method 2: Context Manager (Isolated Tests)
```python
def test_with_context_manager():
    lm = DummyLM([{"answer": "isolated response"}])
    with dspy.context(lm=lm):
        predictor = dspy.Predict("question -> answer")
        result = predictor(question="Test question")
        assert result.answer == "isolated response"
```

#### Method 3: Pytest Fixtures
```python
@pytest.fixture
def dummy_lm():
    """Reusable DummyLM fixture."""
    lm = DummyLM([{"answer": "fixture response"}])
    with dspy.context(lm=lm):
        yield lm

@pytest.fixture
def configured_predictor(dummy_lm):
    """Fixture with pre-configured predictor."""
    return dspy.Predict("question -> answer")

def test_with_fixtures(configured_predictor):
    result = configured_predictor(question="Any question")
    assert result.answer == "fixture response"
```

## Understanding DummyLM Modes

### Mode 1: List Mode (Sequential Responses)

Cycles through a list of predefined responses in order. Perfect for testing multi-step workflows.

```python
def test_list_mode_sequential():
    responses = [
        {"step": "initialize", "status": "ready"},
        {"step": "process", "status": "running"},
        {"step": "complete", "status": "done"}
    ]
    lm = DummyLM(responses)
    dspy.settings.configure(lm=lm)

    predictor = dspy.Predict("command -> step, status")

    # Each call returns the next response
    assert predictor(command="start").step == "initialize"
    assert predictor(command="run").step == "process"
    assert predictor(command="finish").step == "complete"
    # After exhausting the list, returns default
    assert predictor(command="extra").step == "No more responses"
```

### Mode 2: Dictionary Mode (Content-Based Matching)

Returns specific responses based on substring matching in the prompt.

```python
def test_dictionary_mode_matching():
    response_map = {
        "weather": {"answer": "sunny", "temperature": "72°F"},
        "time": {"answer": "3:00 PM", "timezone": "EST"},
        "location": {"answer": "New York", "country": "USA"}
    }

    lm = DummyLM(response_map)
    dspy.settings.configure(lm=lm)

    predictor = dspy.Predict("query -> answer, temperature, timezone, country")

    # Matches "weather" in the query
    weather_result = predictor(query="What's the weather today?")
    assert weather_result.answer == "sunny"
    assert weather_result.temperature == "72°F"

    # Matches "time" in the query
    time_result = predictor(query="What time is it?")
    assert time_result.answer == "3:00 PM"
    assert time_result.timezone == "EST"
```

### Mode 3: Example-Following Mode

Mimics examples from demonstrations when the input matches exactly.

```python
def test_example_following_mode():
    # Default responses when no example matches
    lm = DummyLM([{"translation": "Bonjour"}], follow_examples=True)
    dspy.settings.configure(lm=lm)

    predictor = dspy.Predict("text -> translation")

    # Add demonstrations
    predictor.demos = [
        dspy.Example(text="Hello", translation="Hola").with_inputs("text"),
        dspy.Example(text="Goodbye", translation="Adiós").with_inputs("text")
    ]

    # Exact match returns demo output
    assert predictor(text="Hello").translation == "Hola"
    assert predictor(text="Goodbye").translation == "Adiós"

    # No match returns default
    assert predictor(text="Welcome").translation == "Bonjour"
```

## Working Examples

### Example 1: Testing a Simple Classification Agent

```python
import pytest
import dspy
from dspy.utils import DummyLM

class SentimentClassifier(dspy.Module):
    """Classifies text sentiment as positive, negative, or neutral."""

    def __init__(self):
        super().__init__()
        self.classifier = dspy.Predict("text -> sentiment, confidence")

    def forward(self, text):
        return self.classifier(text=text)

class TestSentimentClassifier:
    @pytest.fixture
    def classifier(self):
        """Setup classifier with DummyLM."""
        responses = {
            "happy": {"sentiment": "positive", "confidence": "0.95"},
            "sad": {"sentiment": "negative", "confidence": "0.88"},
            "okay": {"sentiment": "neutral", "confidence": "0.75"}
        }
        lm = DummyLM(responses)
        with dspy.context(lm=lm):
            yield SentimentClassifier()

    def test_positive_sentiment(self, classifier):
        result = classifier("I'm so happy today!")
        assert result.sentiment == "positive"
        assert float(result.confidence) > 0.9

    def test_negative_sentiment(self, classifier):
        result = classifier("This makes me sad")
        assert result.sentiment == "negative"
        assert float(result.confidence) > 0.8

    def test_neutral_sentiment(self, classifier):
        result = classifier("It's okay, I guess")
        assert result.sentiment == "neutral"
        assert float(result.confidence) > 0.7
```

### Example 2: Testing Multi-Step Data Processing Pipeline

```python
class DataPipeline(dspy.Module):
    """Multi-step data processing pipeline."""

    def __init__(self):
        super().__init__()
        self.validator = dspy.Predict("data -> is_valid, errors")
        self.processor = dspy.ChainOfThought("data -> processed_data")
        self.summarizer = dspy.Predict("processed_data -> summary, metrics")

    def forward(self, data):
        # Step 1: Validate
        validation = self.validator(data=data)
        if validation.is_valid != "true":
            return validation

        # Step 2: Process
        processed = self.processor(data=data)

        # Step 3: Summarize
        summary = self.summarizer(processed_data=processed.processed_data)
        return summary

def test_data_pipeline_success():
    """Test successful pipeline execution."""
    responses = [
        {"is_valid": "true", "errors": "none"},
        {"reasoning": "Processing data", "processed_data": "cleaned_data_v1"},
        {"summary": "Data successfully processed", "metrics": "100 records"}
    ]

    lm = DummyLM(responses)
    dspy.settings.configure(lm=lm)

    pipeline = DataPipeline()
    result = pipeline(data="raw_input_data")

    assert result.summary == "Data successfully processed"
    assert result.metrics == "100 records"

def test_data_pipeline_validation_failure():
    """Test pipeline handling validation errors."""
    responses = [
        {"is_valid": "false", "errors": "Invalid format"}
    ]

    lm = DummyLM(responses)
    dspy.settings.configure(lm=lm)

    pipeline = DataPipeline()
    result = pipeline(data="bad_data")

    assert result.is_valid == "false"
    assert result.errors == "Invalid format"
```

### Example 3: Testing Async Operations

```python
@pytest.mark.asyncio
async def test_async_agent_workflow():
    """Test async agent with multiple async operations."""

    class AsyncAgent(dspy.Module):
        def __init__(self):
            super().__init__()
            self.analyzer = dspy.ChainOfThought("query -> analysis")
            self.resolver = dspy.Predict("analysis -> solution")

        async def aforward(self, query):
            analysis = await self.analyzer.acall(query=query)
            solution = await self.resolver.acall(analysis=analysis.analysis)
            return solution

    responses = [
        {"reasoning": "Analyzing query", "analysis": "Complex problem identified"},
        {"solution": "Apply algorithm X with parameters Y"}
    ]

    lm = DummyLM(responses)
    with dspy.context(lm=lm):
        agent = AsyncAgent()
        result = await agent.aforward(query="Solve optimization problem")

        assert result.solution == "Apply algorithm X with parameters Y"

@pytest.mark.asyncio
async def test_parallel_async_calls():
    """Test multiple async calls in parallel."""
    import asyncio

    lm = DummyLM([
        {"result": "Task 1 complete"},
        {"result": "Task 2 complete"},
        {"result": "Task 3 complete"}
    ])

    with dspy.context(lm=lm):
        predictor = dspy.Predict("task -> result")

        tasks = [
            predictor.acall(task="Task 1"),
            predictor.acall(task="Task 2"),
            predictor.acall(task="Task 3")
        ]

        results = await asyncio.gather(*tasks)

        assert len(results) == 3
        assert results[0].result == "Task 1 complete"
        assert results[1].result == "Task 2 complete"
        assert results[2].result == "Task 3 complete"
```

### Example 4: Testing with Complex Pydantic Types

```python
from pydantic import BaseModel
from typing import List
from datetime import datetime

class Document(BaseModel):
    title: str
    content: str
    tags: List[str]
    created_at: datetime

class DocumentProcessor(dspy.Module):
    """Process and categorize documents."""

    class ProcessingSignature(dspy.Signature):
        document: Document = dspy.InputField()
        category: str = dspy.OutputField()
        priority: int = dspy.OutputField()
        summary: str = dspy.OutputField()

    def __init__(self):
        super().__init__()
        self.processor = dspy.Predict(self.ProcessingSignature)

    def forward(self, document: Document):
        return self.processor(document=document)

def test_document_processor():
    """Test processing documents with complex types."""
    lm = DummyLM([{
        "category": "Technical",
        "priority": 1,
        "summary": "API documentation for v2.0"
    }])

    dspy.settings.configure(lm=lm)

    processor = DocumentProcessor()

    doc = Document(
        title="API Guide",
        content="Complete API reference...",
        tags=["api", "documentation", "v2"],
        created_at=datetime(2024, 1, 15, 10, 30)
    )

    result = processor(document=doc)

    assert result.category == "Technical"
    assert result.priority == 1
    assert "API documentation" in result.summary
```

### Example 5: Testing Error Recovery and Retries

```python
class ResilientAgent(dspy.Module):
    """Agent with error handling and retry logic."""

    def __init__(self, max_retries=3):
        super().__init__()
        self.predictor = dspy.Predict("query -> response, status")
        self.max_retries = max_retries

    def forward(self, query):
        for attempt in range(self.max_retries):
            result = self.predictor(query=f"{query} (attempt {attempt + 1})")
            if result.status == "success":
                return result
        return dspy.Prediction(response="Failed after retries", status="error")

def test_resilient_agent_success_first_try():
    """Test successful response on first attempt."""
    lm = DummyLM([{"response": "Data retrieved", "status": "success"}])
    dspy.settings.configure(lm=lm)

    agent = ResilientAgent()
    result = agent(query="Get data")

    assert result.status == "success"
    assert result.response == "Data retrieved"

def test_resilient_agent_retry_then_success():
    """Test retry mechanism with eventual success."""
    responses = [
        {"response": "Network error", "status": "error"},
        {"response": "Timeout", "status": "error"},
        {"response": "Data retrieved", "status": "success"}
    ]

    lm = DummyLM(responses)
    dspy.settings.configure(lm=lm)

    agent = ResilientAgent()
    result = agent(query="Get data")

    assert result.status == "success"
    assert result.response == "Data retrieved"

def test_resilient_agent_max_retries_exceeded():
    """Test behavior when max retries are exceeded."""
    lm = DummyLM([{"response": "Error", "status": "error"}])
    dspy.settings.configure(lm=lm)

    agent = ResilientAgent(max_retries=2)
    result = agent(query="Get data")

    assert result.status == "error"
    assert result.response == "Failed after retries"
```

### Example 6: Testing Stateful Agents

```python
class StatefulConversationAgent(dspy.Module):
    """Agent that maintains conversation history."""

    def __init__(self):
        super().__init__()
        self.responder = dspy.Predict("message, history -> response")
        self.conversation_history = []

    def forward(self, message):
        history_str = " | ".join(self.conversation_history[-3:])  # Last 3 messages
        result = self.responder(message=message, history=history_str)

        self.conversation_history.append(f"User: {message}")
        self.conversation_history.append(f"Agent: {result.response}")

        return result

class TestStatefulConversation:
    def test_conversation_flow(self):
        """Test maintaining conversation context."""
        responses = [
            {"response": "Hello! How can I help you?"},
            {"response": "Python is a great language for beginners."},
            {"response": "Yes, I remember you asked about Python."}
        ]

        lm = DummyLM(responses)
        dspy.settings.configure(lm=lm)

        agent = StatefulConversationAgent()

        # First interaction
        result1 = agent("Hello")
        assert "Hello" in result1.response
        assert len(agent.conversation_history) == 2

        # Second interaction
        result2 = agent("Tell me about Python")
        assert "Python" in result2.response
        assert len(agent.conversation_history) == 4

        # Third interaction - should have context
        result3 = agent("What did I ask about?")
        assert "Python" in result3.response
        assert len(agent.conversation_history) == 6
```

### Example 7: Testing with Parameterized Tests

```python
class PriorityRouter(dspy.Module):
    """Routes tasks based on priority keywords."""

    def __init__(self):
        super().__init__()
        self.router = dspy.Predict("task -> priority, assigned_team")

    def forward(self, task):
        return self.router(task=task)

@pytest.mark.parametrize("task_description,expected_priority,expected_team", [
    ("URGENT: Server down", "critical", "ops"),
    ("Bug in payment system", "high", "engineering"),
    ("Update documentation", "low", "docs"),
    ("Feature request: dark mode", "medium", "product"),
])
def test_priority_routing(task_description, expected_priority, expected_team):
    """Test task routing with various priority levels."""

    response_map = {
        "URGENT": {"priority": "critical", "assigned_team": "ops"},
        "Bug": {"priority": "high", "assigned_team": "engineering"},
        "documentation": {"priority": "low", "assigned_team": "docs"},
        "Feature": {"priority": "medium", "assigned_team": "product"},
    }

    lm = DummyLM(response_map)
    dspy.settings.configure(lm=lm)

    router = PriorityRouter()
    result = router(task=task_description)

    assert result.priority == expected_priority
    assert result.assigned_team == expected_team
```

## Best Practices

### 1. Test Organization

```python
# conftest.py - Shared fixtures
import pytest
import dspy
from dspy.utils import DummyLM

@pytest.fixture(scope="function")
def reset_dspy():
    """Reset DSPy settings after each test."""
    yield
    dspy.settings.configure(lm=None)

@pytest.fixture
def standard_dummy_lm():
    """Standard DummyLM for common test cases."""
    return DummyLM([{"result": "success"}])

@pytest.fixture
def configured_dspy(standard_dummy_lm):
    """Pre-configured DSPy context."""
    with dspy.context(lm=standard_dummy_lm):
        yield
```

### 2. Response Data Management

```python
# test_responses.py - Centralized response definitions
class TestResponses:
    """Centralized test response definitions."""

    CLASSIFICATION_RESPONSES = {
        "positive": {"sentiment": "positive", "confidence": "0.95"},
        "negative": {"sentiment": "negative", "confidence": "0.88"},
        "neutral": {"sentiment": "neutral", "confidence": "0.75"}
    }

    WORKFLOW_RESPONSES = [
        {"stage": "init", "status": "ready"},
        {"stage": "process", "status": "running"},
        {"stage": "complete", "status": "done"}
    ]

    ERROR_RESPONSES = [
        {"error": "Invalid input", "code": "400"},
        {"error": "Not found", "code": "404"},
        {"error": "Server error", "code": "500"}
    ]

# Usage in tests
def test_with_centralized_responses():
    lm = DummyLM(TestResponses.CLASSIFICATION_RESPONSES)
    dspy.settings.configure(lm=lm)
    # ... rest of test
```

### 3. Testing Best Practices

1. **Isolate Tests**: Use context managers or fixtures to ensure test isolation
2. **Predictable Responses**: Define clear, predictable response patterns
3. **Test Edge Cases**: Include tests for empty responses, errors, and boundary conditions
4. **Document Intent**: Use descriptive test names and comments
5. **Avoid Over-Mocking**: Test actual DSPy behavior, not just mocked responses

### 4. Performance Considerations

```python
@pytest.fixture(scope="module")
def module_scoped_lm():
    """Module-scoped LM for tests that can share configuration."""
    return DummyLM([{"result": "shared"}])

@pytest.mark.performance
def test_large_batch_processing():
    """Test processing large batches efficiently."""
    # Pre-generate responses for large batches
    responses = [{"id": i, "processed": True} for i in range(1000)]
    lm = DummyLM(responses)

    with dspy.context(lm=lm):
        predictor = dspy.Predict("item -> id, processed")

        results = []
        for i in range(1000):
            result = predictor(item=f"Item {i}")
            results.append(result)

        assert len(results) == 1000
        assert all(r.processed for r in results)
```

## Troubleshooting

### Common Issues and Solutions

#### Issue 1: "No more responses" Error
**Problem**: DummyLM returns "No more responses" unexpectedly.

**Solution**:
```python
def test_avoid_exhaustion():
    # BAD: Limited responses for multiple calls
    lm = DummyLM([{"answer": "one"}])  # Only one response!

    # GOOD: Sufficient responses or use dictionary mode
    lm = DummyLM([
        {"answer": "one"},
        {"answer": "two"},
        {"answer": "three"}
    ])

    # BETTER: Use dictionary mode for repeated patterns
    lm = DummyLM({"question": {"answer": "consistent"}})
```

#### Issue 2: Adapter Mismatch
**Problem**: Output format doesn't match expected structure.

**Solution**:
```python
def test_with_correct_adapter():
    from dspy.adapters import JSONAdapter

    # Ensure adapter consistency
    lm = DummyLM([{"field": "value"}], adapter=JSONAdapter())
    dspy.settings.configure(lm=lm, adapter=JSONAdapter())

    predictor = dspy.Predict("input -> field")
    result = predictor(input="test")
    assert result.field == "value"
```

#### Issue 3: Async Test Failures
**Problem**: Async tests fail or hang.

**Solution**:
```python
@pytest.mark.asyncio
async def test_async_properly():
    # Ensure proper async context
    lm = DummyLM([{"result": "async_value"}])

    async with dspy.context(lm=lm):  # Use async context if available
        predictor = dspy.Predict("query -> result")
        result = await predictor.acall(query="test")
        assert result.result == "async_value"
```

#### Issue 4: Dictionary Mode Not Matching
**Problem**: Dictionary mode doesn't return expected responses.

**Solution**:
```python
def test_dictionary_matching():
    # Use substrings that will definitely match
    response_map = {
        "weather": {"answer": "sunny"},  # Matches any prompt containing "weather"
        "time": {"answer": "noon"}
    }

    lm = DummyLM(response_map)
    dspy.settings.configure(lm=lm)

    predictor = dspy.Predict("question -> answer")

    # These will match:
    assert predictor(question="What's the weather?").answer == "sunny"
    assert predictor(question="Tell me about weather").answer == "sunny"

    # This won't match any key:
    result = predictor(question="Hello")
    assert result.answer == "No more responses"
```

#### Issue 5: Example-Following Mode Not Working
**Problem**: follow_examples=True doesn't use demonstration outputs.

**Solution**:
```python
def test_example_following_correct():
    # Must set follow_examples=True
    lm = DummyLM([{"default": "fallback"}], follow_examples=True)
    dspy.settings.configure(lm=lm)

    predictor = dspy.Predict("input -> output")

    # Demos must be properly formatted
    predictor.demos = [
        dspy.Example(input="exact_match", output="demo_output").with_inputs("input")
    ]

    # Input must match demo input exactly
    result = predictor(input="exact_match")  # Matches!
    assert result.output == "demo_output"

    result2 = predictor(input="close_match")  # Doesn't match
    assert result2.output != "demo_output"
```

### Debugging Tips

```python
def test_with_debugging():
    """Example showing debugging techniques."""

    # 1. Inspect LM history
    lm = DummyLM([{"answer": "test"}])
    dspy.settings.configure(lm=lm)

    predictor = dspy.Predict("question -> answer")
    result = predictor(question="What?")

    # Check what was sent to the LM
    print(lm.history[-1])  # Last call details

    # 2. Verify response structure
    assert hasattr(result, 'answer'), f"Missing 'answer' field. Got: {result._store}"

    # 3. Check intermediate steps in multi-step workflows
    class DebugWorkflow(dspy.Module):
        def __init__(self):
            super().__init__()
            self.step1 = dspy.Predict("input -> intermediate")
            self.step2 = dspy.Predict("intermediate -> output")

        def forward(self, input_data):
            intermediate = self.step1(input=input_data)
            print(f"Intermediate result: {intermediate.intermediate}")

            final = self.step2(intermediate=intermediate.intermediate)
            return final
```

## Summary

DummyLM provides a powerful, deterministic way to unit test DSPy applications. By following these patterns and best practices, you can create comprehensive test suites that:

- Run quickly without API calls
- Provide predictable, repeatable results
- Cover edge cases and error scenarios
- Support both sync and async workflows
- Handle complex data types and multi-step processes

Remember to:
- Choose the appropriate DummyLM mode for your test scenario
- Isolate tests using context managers or fixtures
- Provide sufficient responses to avoid exhaustion
- Test both success and failure paths
- Use parameterized tests for comprehensive coverage

For more information, refer to the official DSPy documentation and the source code at `dspy/utils/dummies.py`.

