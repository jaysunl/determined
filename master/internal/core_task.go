package internal

import (
	"github.com/labstack/echo/v4"

	"github.com/determined-ai/determined/master/internal/context"
	expauth "github.com/determined-ai/determined/master/internal/experiment"
	"github.com/determined-ai/determined/master/internal/sproto"
)

func (m *Master) getTasks(c echo.Context) (interface{}, error) {
	summary, err := m.rm.GetAllocationSummaries(m.system, sproto.GetAllocationSummaries{})
	if err != nil {
		return nil, err
	}

	curUser := c.(*context.DetContext).MustGetUser()
	for allocationID := range summary {
		isExp, exp, err := expFromAllocationID(m, allocationID)
		if err != nil {
			return nil, err
		}
		if !isExp {
			continue // TODO(nick) add other task type auth checking.
		}

		if ok, err := expauth.AuthZProvider.Get().CanGetExperiment(curUser, exp); err != nil {
			return nil, err
		} else if !ok {
			delete(summary, allocationID)
		}
	}
	return summary, nil
}
