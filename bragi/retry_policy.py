"""Central quality-first retry budgets.

Attempt counts include the initial operation. Seven attempts therefore permit
six retries before the operation is considered exhausted.
"""

PROVIDER_MAX_ATTEMPTS = 7
MODEL_OUTPUT_MAX_ATTEMPTS = 7
DEFERRED_WORK_MAX_ATTEMPTS = 7
