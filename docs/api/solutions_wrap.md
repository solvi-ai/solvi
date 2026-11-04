# `solvi.solutions.wrap`

`solvi.wrap` configures explicit external-agent decision points. Disabled calls preserve the base call and result; enabled overrides require a strict declared expected-loss reduction, evidence and a durable journal record before return. Internal calls on the original agent are not automatically intercepted. Callback purity and expected-loss validity remain owner contracts; this is not a sandbox or welfare guarantee.

The module also provides training-only paired component selection over predeclared configurations, with incremental score/error records. It does not tune a holdout or certify utility gains. See [external agent wrap](../external-agent-wrap.md).

::: solvi.solutions.wrap
