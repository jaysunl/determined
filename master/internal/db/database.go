package db

import (
	"context"
	"time"

	"github.com/determined-ai/determined/master/internal/sproto"

	"github.com/google/uuid"
	"github.com/pkg/errors"

	"github.com/determined-ai/determined/master/internal/api"
	"github.com/determined-ai/determined/master/internal/lttb"
	"github.com/determined-ai/determined/master/pkg/model"
	"github.com/determined-ai/determined/master/pkg/schemas/expconf"
	"github.com/determined-ai/determined/proto/pkg/apiv1"
	"github.com/determined-ai/determined/proto/pkg/trialv1"
)

// DB is an interface for _all_ the functionality packed into the DB.
type DB interface {
	StartUserSession(user *model.User) (string, error)
	UserByToken(token string) (*model.User, *model.UserSession, error)
	DeleteUserSessionByID(sessionID model.SessionID) error
	UserByUsername(username string) (*model.User, error)
	AddUser(user *model.User, ug *model.AgentUserGroup) error
	UpdateUser(updated *model.User, toUpdate []string, ug *model.AgentUserGroup) error
	UpdateUsername(userID *model.UserID, newUsername string) error
	UserList() (values []model.FullUser, err error)
	UserByID(userID model.UserID) (*model.FullUser, error)
	AgentUserGroup(userID model.UserID) (*model.AgentUserGroup, error)
	Migrate(migrationURL string) error
	Close() error
	GetClusterID() (string, error)
	ExperimentWithTrialSummariesRaw(id int) ([]byte, error)
	ExperimentWithSummaryMetricsRaw(id int) ([]byte, error)
	CheckExperimentExists(id int) (bool, error)
	CheckTrialExists(id int) (bool, error)
	TrialExperimentAndRequestID(id int) (int, model.RequestID, error)
	ExperimentCheckpointsRaw(id int, numBest *int) ([]byte, error)
	ExperimentConfigRaw(id int) ([]byte, error)
	ExperimentConfigByTrialsRaw(trialIDs []int) ([]byte, error)
	ExperimentRaw(id int) ([]byte, error)
	ExperimentListRaw(
		skipArchived bool, username string, limit, offset int,
	) ([]byte, error)
	ExperimentDescriptorsRaw(skipArchived, skipInactive bool) ([]byte, error)
	ExperimentDescriptorsRawForUser(skipArchived, skipInactive bool,
		username string) ([]byte, error)
	AddExperiment(experiment *model.Experiment) error
	ExperimentByID(id int) (*model.Experiment, error)
	LegacyExperimentConfigByID(
		id int,
	) (expconf.LegacyConfig, error)
	ExperimentWithoutConfigByID(id int) (*model.Experiment, error)
	ExperimentIDByTrialID(trialID int) (int, error)
	ExperimentByTrialID(id int) (*model.Experiment, error)
	NonTerminalExperiments() ([]*model.Experiment, error)
	TerminateExperimentInRestart(id int, state model.State) error
	SaveExperimentConfig(experiment *model.Experiment) error
	SaveExperimentState(experiment *model.Experiment) error
	SaveExperimentArchiveStatus(experiment *model.Experiment) error
	DeleteExperiment(id int) error
	ExperimentHasCheckpointsInRegistry(id int) (bool, error)
	SaveExperimentProgress(id int, progress *float64) error
	ExperimentConfig(id int) (expconf.ExperimentConfig, error)
	ExperimentTotalStepTime(id int) (float64, error)
	ExperimentNumTrials(id int) (int64, error)
	ExperimentTrialIDs(expID int) ([]int, error)
	ExperimentNumSteps(id int) (int64, error)
	ExperimentModelDefinitionRaw(id int) ([]byte, error)
	ExperimentCheckpointsToGCRaw(
		id int,
		experimentBest, trialBest, trialLatest int,
		delete bool,
	) ([]byte, error)
	AddTrial(trial *model.Trial) error
	TrialByID(id int) (*model.Trial, error)
	TrialByExperimentAndRequestID(
		experimentID int, requestID model.RequestID,
	) (*model.Trial, error)
	UpdateTrial(id int, newState model.State) error
	UpdateTrialRunnerState(id int, state string) error
	UpdateTrialRunnerMetadata(id int, md *trialv1.TrialRunnerMetadata) error
	RollBackTrial(id, totalBatches int) error
	TrialDetailsRaw(id int) ([]byte, error)
	AddStep(step *model.Step) error
	AddNoOpStep(step *model.Step) error
	AddTask(jobType model.JobType, jobTypeFkID int, taskID sproto.TaskID) error
	CompleteTask(jobType model.JobType, jobTypeFkID int, taskID sproto.TaskID) error
	EndTasks(jobType model.JobType, jobTypeFkID int) error
	TrialRunIDAndRestarts(trialID int) (int, int, error)
	UpdateTrialRunID(id, runID int) error
	UpdateTrialRestarts(id, restarts int) error
	StepByTotalBatches(trialID, totalBatches int) (*model.Step, error)
	UpdateStep(
		trialID, totalBatches int, newState model.State, metrics model.JSONObj) error
	AddTrainingMetrics(ctx context.Context, m *trialv1.TrainingMetrics) error
	AddValidationMetrics(
		ctx context.Context, m *trialv1.ValidationMetrics,
	) error
	AddCheckpointMetadata(
		ctx context.Context, m *trialv1.CheckpointMetadata,
	) error
	AddValidation(validation *model.Validation) error
	ValidationByTotalBatches(trialID, totalBatches int) (*model.Validation, error)
	UpdateValidation(
		trialID, totalBatches int, newState model.State, metrics model.JSONObj,
	) error
	AddCheckpoint(checkpoint *model.Checkpoint) error
	CheckpointByTotalBatches(trialID, totalBatches int) (*model.Checkpoint, error)
	CheckpointByUUID(id uuid.UUID) (*model.Checkpoint, error)
	LatestCheckpointForTrial(trialID int) (*model.Checkpoint, error)
	UpdateCheckpoint(
		trialID, totalBatches int,
		newCheckpoint model.Checkpoint,
	) error
	UpdateCheckpointMetadata(checkpoint *model.Checkpoint) error
	PeriodicTelemetryInfo() ([]byte, error)
	AddAuthTokenKeypair(tokenKeypair *model.AuthTokenKeypair) error
	AuthTokenKeypair() (*model.AuthTokenKeypair, error)
	TrialState(trialID int) (model.State, error)
	TrialStatus(trialID int) (model.State, *time.Time, error)
	Query(queryName string, v interface{}, params ...interface{}) error
	QueryF(
		queryName string, args []interface{}, v interface{}, params ...interface{}) error
	RawQuery(queryName string, params ...interface{}) ([]byte, error)
	SetTrialBestValidation(id int) error
	UpdateResourceAllocationAggregation() error
	TemplateList() (values []model.Template, err error)
	TemplateByName(name string) (value model.Template, err error)
	UpsertTemplate(tpl *model.Template) error
	DeleteTemplate(name string) error
	InsertTrialProfilerMetricsBatch(
		values []float32, batches []int32, timestamps []time.Time, labels []byte,
	) error
	GetTrialProfilerMetricsBatches(
		labelsJSON []byte, offset, limit int,
	) (model.TrialProfilerMetricsBatchBatch, error)
	ExperimentLabelUsage() (labelUsage map[string]int, err error)
	GetExperimentStatus(experimentID int) (state model.State, progress float64,
		err error)
	MetricNames(experimentID int, sStartTime time.Time, vStartTime time.Time) (
		training []string, validation []string, sEndTime time.Time, vEndTime time.Time, err error)
	TrainingMetricBatches(experimentID int, metricName string, startTime time.Time) (
		batches []int32, endTime time.Time, err error)
	ValidationMetricBatches(experimentID int, metricName string, startTime time.Time) (
		batches []int32, endTime time.Time, err error)
	TrainingTrialsSnapshot(experimentID int, minBatches int, maxBatches int,
		metricName string, startTime time.Time) (trials []*apiv1.TrialsSnapshotResponse_Trial,
		endTime time.Time, err error)
	ValidationTrialsSnapshot(experimentID int, minBatches int, maxBatches int,
		metricName string, startTime time.Time) (trials []*apiv1.TrialsSnapshotResponse_Trial,
		endTime time.Time, err error)
	TopTrialsByMetric(experimentID int, maxTrials int, metric string,
		smallerIsBetter bool) (trials []int32, err error)
	TopTrialsByTrainingLength(experimentID int, maxTrials int, metric string,
		smallerIsBetter bool) (trials []int32, err error)
	TrainingMetricsSeries(trialID int32, startTime time.Time, metricName string,
		startBatches int, endBatches int) (metricSeries []lttb.Point, maxEndTime time.Time,
		err error)
	ValidationMetricsSeries(trialID int32, startTime time.Time, metricName string,
		startBatches int, endBatches int) (metricSeries []lttb.Point, maxEndTime time.Time,
		err error)
	FetchHPImportanceTrainingData(experimentID int, metric string) (
		map[int][]model.HPImportanceTrialData, error)
	FetchHPImportanceValidationData(experimentID int, metric string) (
		map[int][]model.HPImportanceTrialData, error)
	GetHPImportance(experimentID int) (result model.ExperimentHPImportance, err error)
	SetHPImportance(experimentID int, value model.ExperimentHPImportance) error
	GetPartialHPImportance() ([]int, []model.ExperimentHPImportance, error)
	ExperimentBestSearcherValidation(id int) (float32, error)
	StartTaskSession(taskID string) (string, error)
	TaskSessionByToken(token string) (*model.TaskSession, error)
	DeleteTaskSessionByTaskID(taskID string) error
	ExperimentSnapshot(experimentID int) ([]byte, int, error)
	SaveSnapshot(
		experimentID int, version int, experimentSnapshot []byte,
	) error
	DeleteSnapshotsForExperiment(experimentID int) error
	DeleteSnapshotsForTerminalExperiments() error
	QueryProto(queryName string, v interface{}, args ...interface{}) error
	QueryProtof(
		queryName string, args []interface{}, v interface{}, params ...interface{}) error
	TrialLogs(
		trialID, limit int, fs []api.Filter, order apiv1.OrderBy, followState interface{},
	) ([]*model.TrialLog, interface{}, error)
	AddTrialLogs(logs []*model.TrialLog) error
	DeleteTrialLogs(ids []int) error
	TrialLogsCount(trialID int, fs []api.Filter) (int, error)
	TrialLogsFields(trialID int) (*apiv1.TrialLogsFieldsResponse, error)
}

// ErrNotFound is returned if nothing is found.
var ErrNotFound = errors.New("not found")

// ErrTooManyRowsAffected is returned if too many rows are affected.
var ErrTooManyRowsAffected = errors.New("too many rows are affected")

// ErrDuplicateRecord is returned when trying to create a row that already exists.
var ErrDuplicateRecord = errors.New("row already exists")
