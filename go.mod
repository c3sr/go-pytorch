module github.com/c3sr/go-pytorch

go 1.15

replace (
	github.com/coreos/bbolt => go.etcd.io/bbolt v1.3.5
	github.com/jaegertracing/jaeger => github.com/uber/jaeger v1.22.0
	github.com/uber/jaeger => github.com/jaegertracing/jaeger v1.22.0
	google.golang.org/grpc => google.golang.org/grpc v1.29.1
)

require (
	github.com/Masterminds/semver v1.5.0 // indirect
	github.com/VividCortex/ewma v1.1.1 // indirect
	github.com/VividCortex/robustly v0.0.0-20210119222408-48da771af5f6 // indirect
	github.com/aarondl/tpl v0.0.0-20180717141031-b5afe9b3122c // indirect
	github.com/anthonynsimon/bild v0.13.0
	github.com/apache/arrow/go/arrow v0.0.0-20201229220542-30ce2eb5d4dc // indirect
	github.com/benesch/cgosymbolizer v0.0.0-20190515212042-bec6fe6e597b
	github.com/c3sr/archive v1.0.0 // indirect
	github.com/c3sr/cmd v1.0.0 // indirect
	github.com/c3sr/config v1.0.1
	github.com/c3sr/database v1.0.0 // indirect
	github.com/c3sr/dlframework v1.0.1
	github.com/c3sr/downloadmanager v1.0.0
	github.com/c3sr/go-cupti v1.0.1
	github.com/c3sr/grpc v1.0.0 // indirect
	github.com/c3sr/image v1.0.0 // indirect
	github.com/c3sr/logger v1.0.1
	github.com/c3sr/monitoring v1.0.0 // indirect
	github.com/c3sr/nvidia-smi v1.0.0
	github.com/c3sr/parallel v1.0.1 // indirect
	github.com/c3sr/pipeline v1.0.0 // indirect
	github.com/c3sr/registry v1.0.0 // indirect
	github.com/c3sr/tracer v1.0.0
	github.com/c3sr/web v1.0.1 // indirect
	github.com/cheggaaa/pb v1.0.29 // indirect
	github.com/cockroachdb/cmux v0.0.0-20170110192607-30d10be49292 // indirect
	github.com/facebookgo/freeport v0.0.0-20150612182905-d4adf43b75b9 // indirect
	github.com/glendc/go-external-ip v0.0.0-20200601212049-c872357d968e // indirect
	github.com/go-openapi/errors v0.20.0 // indirect
	github.com/go-openapi/runtime v0.19.26 // indirect
	github.com/go-openapi/validate v0.20.2 // indirect
	github.com/golang/protobuf v1.5.1 // indirect
	github.com/gorilla/schema v1.2.0 // indirect
	github.com/gorilla/sessions v1.2.1 // indirect
	github.com/h2non/filetype v1.1.1 // indirect
	github.com/jinzhu/copier v0.2.8 // indirect
	github.com/justinas/nosurf v1.1.1 // indirect
	github.com/k0kubun/pp/v3 v3.0.7
	github.com/klauspost/shutdown2 v1.1.0 // indirect
	github.com/labstack/echo v3.3.10+incompatible // indirect
	github.com/levigross/grequests v0.0.0-20190908174114-253788527a1a // indirect
	github.com/olekukonko/tablewriter v0.0.5 // indirect
	github.com/oliamb/cutter v0.2.2 // indirect
	github.com/opentracing/opentracing-go v1.2.0
	github.com/oxtoacart/bpool v0.0.0-20190530202638-03653db5a59c // indirect
	github.com/pkg/errors v0.9.1
	github.com/sirupsen/logrus v1.8.1
	github.com/unknwon/com v1.0.1
	github.com/volatiletech/authboss v2.4.1+incompatible // indirect
	github.com/volatiletech/authboss-clientstate v0.0.0-20200826024349-8d4e74078241 // indirect
	github.com/volatiletech/authboss/v3 v3.0.3 // indirect
	go4.org/unsafe/assume-no-moving-gc v0.0.0-20201222180813-1025295fd063 // indirect
	golang.org/x/crypto v0.0.0-20210317152858-513c2a44f670 // indirect
	golang.org/x/oauth2 v0.0.0-20210313182246-cd4f82c27b84 // indirect
	google.golang.org/genproto v0.0.0-20210318145829-90b20ab00860 // indirect
	gorgonia.org/tensor v0.9.14
)
