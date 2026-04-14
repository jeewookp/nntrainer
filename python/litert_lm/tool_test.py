# Copyright 2026 The ODML Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Sequence

from absl.testing import absltest
from absl.testing import parameterized

import litert_lm


# Helper functions for parameterized tests
def multiply(a: int, b: int) -> int:
  """Multiplies two integers.

  Args:
      a: The first integer.
      b: The second integer.
  """
  return a * b


def greet(name: str, greeting: str = "Hello") -> str:
  """Greets a person."""
  return f"{greeting}, {name}!"


def get_weather(location: str):
  """Gets weather for a location.

  Args:
      location (str): The name of the city.
  """
  return f"Weather in {location} is sunny."


class ToolTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="basic_function",
          func=multiply,
          expected_desc={
              "type": "function",
              "function": {
                  "name": "multiply",
                  "description": "Multiplies two integers.",
                  "parameters": {
                      "type": "object",
                      "properties": {
                          "a": {
                              "type": "integer",
                              "description": "The first integer.",
                          },
                          "b": {
                              "type": "integer",
                              "description": "The second integer.",
                          },
                      },
                      "required": ["a", "b"],
                  },
              },
          },
          execute_cases=[({"a": 2, "b": 3}, 6)],
      ),
      dict(
          testcase_name="function_with_defaults",
          func=greet,
          expected_name="greet",
          expected_required=["name"],
          execute_cases=[
              ({"name": "Alice"}, "Hello, Alice!"),
              ({"name": "Bob", "greeting": "Hi"}, "Hi, Bob!"),
          ],
      ),
      dict(
          testcase_name="type_hints_in_docstring",
          func=get_weather,
          expected_properties={
              "location": {
                  "type": "string",
                  "description": "The name of the city.",
              }
          },
          execute_cases=[
              ({"location": "London"}, "Weather in London is sunny.")
          ],
      ),
  )
  def test_tool_from_function_cases(
      self,
      func,
      execute_cases,
      expected_desc=None,
      expected_name=None,
      expected_required=None,
      expected_properties=None,
  ):
    tool = litert_lm.tool_from_function(func)
    self.assertIsInstance(tool, litert_lm.Tool)
    desc = tool.get_tool_description()

    if expected_desc is not None:
      self.assertEqual(desc, expected_desc)
    if expected_name is not None:
      self.assertEqual(desc["function"]["name"], expected_name)
    if expected_required is not None:
      self.assertEqual(
          desc["function"]["parameters"]["required"], expected_required
      )
    if expected_properties is not None:
      self.assertEqual(
          desc["function"]["parameters"]["properties"], expected_properties
      )

    for args, expected_result in execute_cases:
      result = tool.execute(args)
      self.assertEqual(result, expected_result)

  @parameterized.named_parameters(
      ("list", list),
      ("sequence", Sequence),
  )
  def test_tool_from_function_array(self, type_hint):

    def product(numbers: type_hint[float]) -> float:
      """Calculates the product of a list of numbers.

      Args:
          numbers: List of numbers to multiply.
      """
      res = 1.0
      for n in numbers:
        res *= n
      return res

    tool = litert_lm.tool_from_function(product)
    desc = tool.get_tool_description()

    self.assertEqual(desc["function"]["name"], "product")
    expected_properties = {
        "numbers": {
            "type": "array",
            "items": {"type": "number"},
            "description": "List of numbers to multiply.",
        }
    }
    self.assertEqual(
        desc["function"]["parameters"]["properties"], expected_properties
    )
    self.assertEqual(tool.execute({"numbers": [2.0, 3.5]}), 7.0)


if __name__ == "__main__":
  absltest.main()
