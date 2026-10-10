# Security Policy

## Supported versions

Security fixes are released for the latest minor version of skforecast (for
example, 0.25.x). Please upgrade to the latest release before reporting an
issue, since it may already be fixed.

| Version               | Supported |
|:----------------------|:---------:|
| Latest minor release  | Yes       |
| Older releases        | No        |

## Reporting a vulnerability

Please do not report security vulnerabilities through public GitHub issues,
discussions or pull requests.

Report them privately through one of these channels:

- GitHub private vulnerability reporting: go to the
  [Security tab](https://github.com/skforecast/skforecast/security) of the
  repository and click **Report a vulnerability**.
- Email to the core developers: j.amatrodrigo@gmail.com and
  javier.escobar.ortiz@gmail.com.

Include as much of the following as you can:

- The affected version of skforecast and of its dependencies.
- A description of the vulnerability and its impact.
- Steps or code to reproduce it.
- Any known workaround.

We aim to acknowledge your report within 7 days and will keep you informed
while we work on a fix. Once it is released, we will publish a security
advisory and credit you, unless you prefer to remain anonymous.

## Loading saved forecasters

`save_forecaster` and `load_forecaster` serialize forecasters with joblib,
pickle or cloudpickle. Loading one of these files can execute arbitrary code,
which is how these formats work in Python and not a vulnerability in
skforecast. Only load forecasters from sources you trust.
