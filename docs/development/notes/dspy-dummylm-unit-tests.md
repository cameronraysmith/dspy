# DSPy pytest unit tests with DummyLM 
<optimized_prompt>
<context>
  Help developers understand and implement comprehensive pytest unit testing strategies using DummyLM for Python libraries that create AI agent workflows with DSPy modules.

  Core Rules and Constraints

  - Focus exclusively on DummyLM usage patterns, not real LM testing
  - Provide working, executable code examples
  - Emphasize deterministic, fast, cost-free testing
  - Cover all three DummyLM operational modes
  - Include both sync and async testing patterns
  - Do not recommend integration testing approaches (focus on unit tests only)

  DSPy Testing Context

  1. List Mode: Cycles through predefined responses
  lm = DummyLM([{"answer": "red"}, {"answer": "blue"}])
  2. Dictionary Mode: Returns responses based on prompt content matching
  lm = DummyLM({"What color is the sky?": {"answer": "blue"}})
  3. Example-Following Mode: Mimics examples from demonstrations
  lm = DummyLM([{"answer": "red"}], follow_examples=True)
  Step-by-Step Testing Implementation Process

  Step 1: Basic Test Setup Pattern

  Show how to configure DummyLM with dspy.settings.configure()

  Step 2: Testing DSPy Modules

  Demonstrate testing custom modules that inherit from dspy.Module

  Step 3: Multi-Step Workflow Testing

  Cover complex agent workflows with multiple LM interactions

  Step 4: Async Testing Patterns

  Include async/await testing for modern DSPy applications

  Step 5: Advanced Response Patterns

  Show sophisticated DummyLM response configurations
</context>

## Required Examples

<example>
  def test_basic_agent_response():
      # Configure DummyLM with expected responses
      lm = DummyLM([{"reasoning": "analyze the request", "action": "process_data"}])
      dspy.settings.configure(lm=lm)

  # Test the agent
  agent = MyDSPyAgent()
  result = agent.process("test input")

  # Assertions
  assert result.action == "process_data"
  assert "analyze" in result.reasoning
</example>

<example>
  **Multi-Step Workflow Test**
  ```python
  def test_complex_workflow():
      # Multiple responses for chained operations
      lm = DummyLM([
          {"analysis": "Input requires data processing", "next_step": "retrieve"},
          {"data": "Retrieved information X", "summary": "Data processed"},
          {"final_result": "Success", "confidence": "high"}
      ])
      dspy.settings.configure(lm=lm)

      workflow = MyComplexWorkflow()
      result = workflow.execute("complex task")

      assert result.final_result == "Success"
      assert result.confidence == "high"
  agent = MyPriorityAgent()

  urgent_result = agent.classify("urgent task detected")
  assert urgent_result.priority == "high"

  routine_result = agent.classify("routine task detected") 
  assert routine_result.priority == "normal"
</example>

  ## Input Data

  <testing_requirements>
  The developer needs to understand:
  1. How to configure DummyLM for different test scenarios
  2. Patterns for testing DSPy modules with predictable responses
  3. Best practices for test organization and fixtures
  4. How to test both successful and error scenarios
  5. Integration with pytest fixtures and async testing
  6. Performance testing considerations
  </testing_requirements>

  <file_structure_context>
  Relevant DSPy files for understanding:
  - `dspy/utils/dummies.py` - DummyLM implementation
  - `tests/predict/test_chain_of_thought.py` - DSPy's own testing examples
  - `tests/conftest.py` - Test configuration patterns
  - `dspy/utils/__init__.py` - Import patterns for DummyLM
  </file_structure_context>

  ## Task Reminder
  Create a comprehensive, practical guide for using DummyLM to write pytest unit tests for DSPy-based libraries, with working examples and clear implementation patterns.

  Think step-by-step about the most common testing scenarios developers will encounter, then provide detailed guidance with executable code examples.

  ## Output Format

  Structure your response as:

  <testing_guide>
  [Complete implementation guide with sections for setup, patterns, and best practices]
  </testing_guide>

  <code_examples>
  [5-7 working pytest examples covering different scenarios]
  </code_examples>

  <best_practices>
  [Specific recommendations for DummyLM testing patterns]
  </best_practices>

  <troubleshooting>
  [Common issues and solutions when using DummyLM in tests]
  </troubleshooting>
  </optimized_prompt>

