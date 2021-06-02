package pytorch

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"time"

	"github.com/c3sr/tracer"
	opentracing "github.com/opentracing/opentracing-go"
)

type TraceEvent struct {
	Name      string    `json:"name,omitempty"`
	Phase     string    `json:"ph,omitempty"`
	Timestamp float64   `json:"ts,omitempty"`
	Duration  float64   `json:"dur,omitempty"`
	ProcessID string    `json:"pid,omitempty"`
	ThreadID  int64     `json:"tid,omitempty"`
	Start     int64     `json:"-"`
	End       int64     `json:"-"`
	StartTime time.Time `json:"-"`
	EndTime   time.Time `json:"-"`
	Seq       int64     `json:"-"`
}

func (t TraceEvent) ID() string {
	return fmt.Sprintf("%s/%v", t.Name, t.ThreadID)
}

type TraceEvents []TraceEvent

func (t TraceEvents) Len() int      { return len(t) }
func (t TraceEvents) Swap(i, j int) { t[i], t[j] = t[j], t[i] }
func (t TraceEvents) Less(i, j int) bool {
	if t[i].Start == t[j].Start {
		if t[i].End == t[j].End {
			return t[i].Seq > t[j].Seq
		}
		return t[i].End > t[j].End
	}
	return t[i].Start < t[j].Start
}

type Trace struct {
	StartTime   time.Time
	TraceEvents TraceEvents
}

func (t Trace) Len() int           { return t.TraceEvents.Len() }
func (t Trace) Swap(i, j int)      { t.TraceEvents.Swap(i, j) }
func (t Trace) Less(i, j int) bool { return t.TraceEvents.Less(i, j) }

func NewTrace(data string, start_time int64) (*Trace, error) {
	trace := new(Trace)
	err := json.Unmarshal([]byte(data), &trace.TraceEvents)
	if err != nil {
		return nil, err
	}
	trace.StartTime = time.Unix(0, start_time)
	for ii, event := range trace.TraceEvents {
		trace.TraceEvents[ii].Start = start_time + int64(event.Timestamp*1000)
		trace.TraceEvents[ii].StartTime = time.Unix(0, trace.TraceEvents[ii].Start)
		trace.TraceEvents[ii].End = start_time + int64(event.Timestamp*1000+event.Duration*1000)
		trace.TraceEvents[ii].EndTime = time.Unix(0, trace.TraceEvents[ii].End)
		trace.TraceEvents[ii].Seq = int64(ii)
	}
	return trace, nil
}

func (event *TraceEvent) Publish(ctx context.Context, lvl tracer.Level, idx int, opts ...opentracing.StartSpanOption) error {
	tags := opentracing.Tags{
		"phase":                event.Phase,
		"process_id":           event.ProcessID,
		"thread_id":            event.ThreadID,
		"layer_sequence_index": idx,
	}
	s, _ := tracer.StartSpanFromContext(
		ctx,
		lvl,
		event.Name,
		opentracing.StartTime(event.StartTime),
		tags,
	)
	if s == nil {
		log.WithField("event_name", event.Name).
			WithField("tags", tags).
			Error("failed to create span from context")
		return nil
	}
	s.FinishWithOptions(opentracing.FinishOptions{
		FinishTime: event.EndTime,
	})
	return nil
}

func (t *Trace) Publish(ctx context.Context, lvl tracer.Level, opts ...opentracing.StartSpanOption) error {
	sort.Sort(t.TraceEvents)
	st, ed, idx := int64(-1), int64(-1), 0
	for _, event := range t.TraceEvents {
		if event.Name == "forward" {
			continue
		}
		if event.Start >= st && event.End <= ed {
			continue
		}
		st, ed = event.Start, event.End
		if err := event.Publish(ctx, lvl, idx, opts...); err != nil {
			return err
		}
		idx++
	}
	return nil
}
