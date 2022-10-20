
.PHONY: examples/nlg-reddit/author-level-dp/env
examples/nlg-reddit/author-level-dp/env:
	conda env create -f examples/nlg-reddit/author-level-dp/environment.yml

.PHONY: examples/nlg-reddit/sample-level-dp/env
examples/nlg-reddit/sample-level-dp/env:
	conda env create -f examples/nlg-reddit/sample-level-dp/environment.yml