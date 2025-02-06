[![FINOS - Incubating](https://cdn.jsdelivr.net/gh/finos/contrib-toolbox@master/images/badge-incubating.svg)](https://finosfoundation.atlassian.net/wiki/display/FINOS/Incubating)

# FINOS DTCC Hackathon 


## Project Name: Generative AI Processing of Corporate Actions


### Project Details


### Team Information
- Joseph Uzubell
- Zack Pan
- Elizabeth Smith

### User Setup Guide
Required Software:
- AWS credentials saved to environment variables: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and AWS_SESSION_TOKEN.
- Python >= 3.11.5

Setup commands:
```
# Create python virtual environment
python3 -m venv testing-env
```
```
# Activate python virtual environment on UNIX or Mac OS
source testing-env/bin/activate
```
```
# Install required dependencies in the testing-env (activate first)
pip install -r requirements.txt
```
```
# Add your AWS credentials to your environment variables
export <YOUR_AWS_ACCESS_KEY_ID>
export <YOUR_AWS_SECRET_ACCESS_KEY>
export <YOUR_AWS_SESSION_TOKEN>
```
```
# Run AI solution that parses a Corporate Actions PDF and returns structured data output
python3 src/parse_corporate_action.py
```


## Using DCO to sign your commits

**All commits** must be signed with a DCO signature to avoid being flagged by the DCO Bot. This means that your commit log message must contain a line that looks like the following one, with your actual name and email address:

```
Signed-off-by: John Doe <john.doe@example.com>
```

Adding the `-s` flag to your `git commit` will add that line automatically. You can also add it manually as part of your commit log message or add it afterwards with `git commit --amend -s`.

See [CONTRIBUTING.md](./.github/CONTRIBUTING.md) for more information

### Helpful DCO Resources
- [Git Tools - Signing Your Work](https://git-scm.com/book/en/v2/Git-Tools-Signing-Your-Work)
- [Signing commits
](https://docs.github.com/en/github/authenticating-to-github/signing-commits)


## License

Copyright 2025 FINOS

Distributed under the [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0).

SPDX-License-Identifier: [Apache-2.0](https://spdx.org/licenses/Apache-2.0)








