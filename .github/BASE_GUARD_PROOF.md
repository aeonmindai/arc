This file exists only to give PR `ci/base-guard-proof` a commit.

It is the deliberate NEGATIVE TEST for the `base-branch` lane added in
`ci/base-must-be-master`: this branch's PR targets that branch rather than
`master`, so the lane MUST fail and take `CI complete` down with it.

A guard that has never been observed failing is the exact fault it exists to
catch. The PR is closed and this branch deleted once the red is recorded.
