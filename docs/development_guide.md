# Development Guide
>Best practices, instructions, troubleshooting, and tips for working as efficiently as possible
with the team and app.

## Table of contents
- [Resources](#resources)
- [Git Workflows](#git-workflows)
   - [Hooks](#hooks)
   - [Branches](#branches)
   - [Commits](#commits)
   - [Pull Requests](#pull-requests)

## Resources

- [Issues / Scrum board (JIRA)](https://gbdpaas.atlassian.net/secure/RapidBoard.jspa?rapidView=62)
- [Docs / Wiki (Confluence)](https://gbdpaas.atlassian.net/wiki/spaces/RP/pages)
- [JavaScript Styleguide](https://github.com/glortho/javascript)

## Git Workflows

### Working on [pyveda](https://github.com/DigitalGlobe/pyveda/) inside [Veda](https://github.com/DigitalGlobe/veda/)

pyveda should've been checked out as a submodule
in Veda when you went through the set-up process.

When ready to make changes to pyveda, do the following:

1. Navigate to the pyveda directory inside the Veda repo.

1. Checkout or create the branch you want to work on. Note that as long as
   you're in the pyveda directory, you'll be working on/with only pyveda
   branches, separate from whatever branch you're on in Veda.

1. Make whatever changes to pyveda code, commit them, and push them up if
   ready.

1. When/if you want to work on Veda code, just `cd` out of the pyveda root
   directory. All code outside that pyveda submodule is part of Veda source
   control.

1. Note that if you're working in a notebook where you've imported pyveda you
   may need to restart the notebook kernel in order to see your pyveda changes.

### Branches

Our standard branching flow goes like this:

1. Pull latest `development`.

1. Branch off of `development`.

1. Name it with a JIRA issue prefix, such as `RPT-1234-implementing-some-feature`.

1. Make frequent, small commits on this branch.

1. Submit a PR for your branch (see below), targeting `development` for merge.

If you don't have a JIRA ticket you're working off of, create one via Slack! `% task fix something`

### Commits and Comments

- Architecture and broad decisions are written up in design **docs**.
- Explain what code does and why in **code comments**.
- Explain why changes in the code were necessary in **commit messages**.
- Give disposable help to reviewers and testers in **comments on the pull request**.

Note: Cross-referencing among the above is encouraged!

*Modified from [here](https://medium.com/square-corner-blog/how-square-writes-commit-messages-8e92fcbf77c9).*

#### A great commit message

```
JIRA-1234 Short (50 chars or less) summary of changes

More detailed explanatory text, if necessary.  Wrap it to
about 72 characters or so.  In some contexts, the first
line is treated as the subject of an email and the rest of
the text as the body.  The blank line separating the
summary from the body is critical (unless you omit the body
entirely); tools like rebase can get confused if you run
the two together.

Further paragraphs come after blank lines.

  - Bullet points are okay, too

  - Typically a hyphen or asterisk is used for the bullet,
    preceded by a single space, with blank lines in
    between, but conventions vary here
```

*Modified from [here](https://git-scm.com/book/en/v2/Distributed-Git-Contributing-to-a-Project).*

### Pull Requests

#### Submitting PRs

When you create a pull request [here](https://github.com/DigitalGlobe/timbr-omni/pulls) or wherever, our template will pop into the content box.

1. Fill in as many of the sections as you can, deleting any that are not relevant

1. Check all the relevant boxes in the "Checklist" section, deleting checkboxes that are not relevant

1. Add one **Assignee**. This is the person you want to test and merge the pr.

1. Optionally add one or more **Reviewers**. These are people who you think might have an important opinion to share, or would appreciate knowing the new functionality is coming.

#### Reviewing PRs

If you are a **Reviewer**:

1. Go to the "Files changed" tab to review the code, adding comments inline with the code.

1. When done with your review, click the "Review changes" button at top right and indicate whether you are approving the PR or requesting changes.

If you are the **Assignee**:

1. Pull down the PR branch and follow the instructions in the PR description to test it.

1. Leave any feedback about general functionality in comments on the PR.

1. Go to the "Files changed" tab to review the code, adding comments inline with the code.

1. When done with your review, click the "Review changes" button at top right and indicate whether you are approving the PR or requesting changes.

1. When you and all reviewers have approved the PR, all checkboxes are checked, and the automated tests came back with success, merge the PR.

