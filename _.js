[
  {
    role: 'system',
    content:
      "You will be given a list of phrases, each of which is a phrase describing an object in the picture. Your task is to determine whether the object in these items is greater than or equal to 5. Please provide the answer in the form of words, each word represents 'yes' or 'no'. 'Yes' means that the object is greater than or equal to 5, and 'no' means less than 5. For example: ['6', 'the number 2', '0', 'the number 8'] should be answered with [yes, no, no, yes].",
  },
  {
    role: 'user',
    content: [
      'the number 6',
      'the number 0',
      'The number 5',
      '8',
      '2',
      '5',
      'The number is in the image',
      'asdasdasda',
    ],
  },
  {
    role: 'assistant',
    content: "['yes', 'no', 'yes', 'yes', 'no', 'yes', 'no', 'no']",
  },
];
