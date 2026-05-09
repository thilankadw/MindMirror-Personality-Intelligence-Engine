-- MindMirror initial schema
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'job_type_enum') THEN
        CREATE TYPE job_type_enum AS ENUM ('SIGNUP_INITIAL', 'WEEKLY_INFERENCE');
    END IF;
END$$;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'job_status_enum') THEN
        CREATE TYPE job_status_enum AS ENUM ('QUEUED', 'RUNNING', 'SUCCEEDED', 'FAILED');
    END IF;
END$$;

CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    hashed_password VARCHAR(255),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    is_active BOOLEAN NOT NULL DEFAULT TRUE
);

CREATE TABLE IF NOT EXISTS jobs (
    job_id UUID PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES users(id),
    job_type job_type_enum NOT NULL,
    week_start_date DATE,
    status job_status_enum NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    finished_at TIMESTAMPTZ,
    error_message TEXT
);

CREATE INDEX IF NOT EXISTS idx_jobs_user_id ON jobs(user_id);

CREATE TABLE IF NOT EXISTS predictions (
    id BIGSERIAL PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES users(id),
    week_start_date DATE NOT NULL,
    model_name VARCHAR(128) NOT NULL,
    prediction_value DOUBLE PRECISION NOT NULL,
    confidence DOUBLE PRECISION,
    model_version VARCHAR(64) NOT NULL,
    feature_schema_version VARCHAR(32) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_predictions_user_week_model UNIQUE (user_id, week_start_date, model_name)
);

CREATE INDEX IF NOT EXISTS idx_predictions_user_week ON predictions(user_id, week_start_date DESC);
