# Assignment 3 for the DataCamp course X-DataScience Master - scikit-learn API

## What we want you to check that you know how to do by doing this assignment:

  - Use Git and GitHub
  - Work with Python files (and not just notebooks!)
  - Do a pull request on a GitHub repository
  - Format your code properly using standard Python conventions
  - Make your code pass tests run automatically on a continuous in
  tegration system (GitHub actions)
  - Understand how to code scikit-learn compatible objects.

## How?

  - Fork the repository by clicking on the `Fork` button on the upper right corner
  - Clone the repository of your fork with: `git clone https://github.com/MYLOGIN/datacamp-assignment-sklearn` (replace MYLOGIN with your GitHub login)
  - Create a branch called `MYLOGIN` using `git checkout -b MYLOGIN`
  - Make the changes to complete the assignment. You have to modify the files that contain `questions` in their name. Do not modify the files that start with `test_`.  - Check locally that your solution meet the test by running `pytest` from the root of the repo. You may need to install `pytest` using `pip` or `conda`.
  - Check the code formating for your solution using `flake8`. You may need to install `flake8` using `pip` or `conda`.
  - Open the pull request on GitHub:
     - Create a commit with `git add -u` and `git commit -m "UP my solution"`
     - Push your branch on your fork: `git push -u origin MYLOGIN`
     - Go to your repo in your browser and click the `Open a PR` button.
  - Keep pushing to your branch until the continuous integration system is green.
  - When it is green notify the instructors on Slack that your done.

# Your mission

- You should implement a scikit-learn estimator for the `KNearestNeighbors` class. This corresponds to implementing the methods `fit`, `predict` and `score` of the class in `sklearn_questions.py`.
- You should implement a scikit-learn cross-validator for the `MonthlySplit` class. This corresponds to implementing the methods `get_n_splits` and `split` of the class in `sklearn_questions.py`.

## Getting Help

If you need help ask on the Slack of the Datacamp course.
