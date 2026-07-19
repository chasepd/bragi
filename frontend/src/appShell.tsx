import React, { Suspense, useCallback, useEffect, useMemo, useState } from "react";
import ReactDOM from "react-dom/client";
import { api, logClientEvent, postJson, setUnauthorizedHandler } from "./api";
import { BrandLockup } from "./brand";
import "./styles.css";

export type CurrentUser = { id: string; username: string; role: string; status: string };
type AuthResponse = { user: CurrentUser };
type BootstrapStatus = { admin_exists: boolean; bootstrap_required: boolean; setup_token_required: boolean };
type AuthSessionResponse = { bootstrap: BootstrapStatus; user: CurrentUser | null };
type SessionState =
  | { status: "checking" }
  | { status: "bootstrap"; message?: string; setupTokenRequired: boolean }
  | { status: "login"; message?: string }
  | { status: "authenticated"; user: CurrentUser }
  | { status: "error"; message: string };

export type AuthenticatedWorkbenchProps = {
  currentUser: CurrentUser;
  onLogout: () => void;
};

export type AuthenticatedWorkbenchLoader = () => Promise<{
  default: React.ComponentType<AuthenticatedWorkbenchProps>;
}>;

const defaultLoadAuthenticatedWorkbench: AuthenticatedWorkbenchLoader = () => import("./authenticatedWorkbench");

export function App({
  loadAuthenticatedWorkbench = defaultLoadAuthenticatedWorkbench
}: {
  loadAuthenticatedWorkbench?: AuthenticatedWorkbenchLoader;
}) {
  return <SessionShell loadAuthenticatedWorkbench={loadAuthenticatedWorkbench} />;
}

function SessionShell({
  loadAuthenticatedWorkbench
}: {
  loadAuthenticatedWorkbench: AuthenticatedWorkbenchLoader;
}) {
  const [session, setSession] = useState<SessionState>({ status: "checking" });
  const [sessionReload, setSessionReload] = useState(0);
  const AuthenticatedWorkbench = useMemo(
    () => React.lazy(loadAuthenticatedWorkbench),
    [loadAuthenticatedWorkbench]
  );

  const markLoggedOut = useCallback((message?: string) => {
    setSession({ status: "login", message });
  }, []);

  const handleAuthenticated = useCallback((user: CurrentUser) => {
    setSession({ status: "authenticated", user });
  }, []);

  useEffect(() => {
    let active = true;
    async function loadSession() {
      try {
        const session = await api<AuthSessionResponse>("/api/auth/session");
        if (!active) return;
        const bootstrap = session.bootstrap;
        if (bootstrap.bootstrap_required) {
          setSession({ status: "bootstrap", setupTokenRequired: bootstrap.setup_token_required });
          return;
        }
        if (session.user) {
          handleAuthenticated(session.user);
          return;
        }
        setSession({ status: "login" });
      } catch (failure) {
        if (!active) return;
        setSession({
          status: "error",
          message: failure instanceof Error ? failure.message : "Could not load session"
        });
      }
    }
    void loadSession();
    return () => {
      active = false;
    };
  }, [handleAuthenticated, sessionReload]);

  useEffect(() => {
    setUnauthorizedHandler(() => markLoggedOut("Session expired. Log in again to continue."));
    return () => setUnauthorizedHandler(null);
  }, [markLoggedOut]);

  const logout = useCallback(async () => {
    try {
      await postJson<{ ok: boolean }>("/api/auth/logout", {});
    } finally {
      markLoggedOut();
    }
  }, [markLoggedOut]);

  if (session.status === "authenticated") {
    return (
      <Suspense fallback={<AuthStatus title="Opening workbench" />}>
        <AuthenticatedWorkbench currentUser={session.user} onLogout={logout} />
      </Suspense>
    );
  }
  if (session.status === "bootstrap") {
    return (
      <AuthPanel
        mode="bootstrap"
        message={session.message}
        setupTokenRequired={session.setupTokenRequired}
        onAuthenticated={handleAuthenticated}
      />
    );
  }
  if (session.status === "login") {
    return (
      <AuthPanel
        mode="login"
        message={session.message}
        onAuthenticated={handleAuthenticated}
      />
    );
  }
  if (session.status === "error") {
    return (
      <main className="auth-shell">
        <div className="auth-scene" aria-hidden="true" />
        <section className="auth-panel" aria-live="polite">
          <BrandLockup />
          <h1>Could not open Bragi</h1>
          <InlineNotice>{session.message}</InlineNotice>
          <button
            type="button"
            onClick={() => {
              setSession({ status: "checking" });
              setSessionReload((value) => value + 1);
            }}
          >
            Retry
          </button>
        </section>
      </main>
    );
  }
  return <AuthStatus title="Opening Bragi" />;
}

function AuthStatus({ title }: { title: string }) {
  return (
    <main className="auth-shell">
      <div className="auth-scene" aria-hidden="true" />
      <section className="auth-panel" aria-live="polite">
        <BrandLockup />
        <h1>{title}</h1>
        <span className="auth-loading-spinner" aria-hidden="true" />
      </section>
    </main>
  );
}

function AuthPanel({
  mode,
  message,
  setupTokenRequired = false,
  onAuthenticated
}: {
  mode: "bootstrap" | "login";
  message?: string;
  setupTokenRequired?: boolean;
  onAuthenticated: (user: CurrentUser) => void;
}) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [setupToken, setSetupToken] = useState("");
  const [error, setError] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const isBootstrap = mode === "bootstrap";
  const title = isBootstrap ? "Create first admin" : "Log in to Bragi";
  const buttonLabel = isBootstrap ? "Create admin" : "Log in";
  const endpoint = isBootstrap ? "/api/bootstrap/admin" : "/api/auth/login";

  async function submit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError("");
    setSubmitting(true);
    try {
      const response = await postJson<AuthResponse>(
        endpoint,
        isBootstrap && setupTokenRequired
          ? { username, password, setup_token: setupToken }
          : { username, password }
      );
      onAuthenticated(response.user);
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : "Authentication failed");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <main className="auth-shell">
      <div className="auth-scene" aria-hidden="true" />
      <form className="auth-panel" onSubmit={submit}>
        <BrandLockup />
        <h1>{title}</h1>
        {message ? <InlineNotice polite>{message}</InlineNotice> : null}
        {error ? <InlineNotice>{error}</InlineNotice> : null}
        <label className="auth-field">
          <span>Username</span>
          <input
            autoComplete="username"
            autoFocus
            value={username}
            onChange={(event) => setUsername(event.currentTarget.value)}
          />
        </label>
        <label className="auth-field">
          <span>Password</span>
          <input
            autoComplete={isBootstrap ? "new-password" : "current-password"}
            minLength={isBootstrap ? 12 : undefined}
            type="password"
            value={password}
            onChange={(event) => setPassword(event.currentTarget.value)}
          />
        </label>
        {isBootstrap && setupTokenRequired ? (
          <label className="auth-field">
            <span>Setup token</span>
            <input
              autoComplete="one-time-code"
              type="password"
              value={setupToken}
              onChange={(event) => setSetupToken(event.currentTarget.value)}
            />
          </label>
        ) : null}
        <button
          type="submit"
          disabled={submitting || !username.trim() || !password || (setupTokenRequired && !setupToken)}
        >
          {submitting ? <span className="auth-button-spinner" aria-hidden="true" /> : null}
          {buttonLabel}
        </button>
      </form>
    </main>
  );
}

function InlineNotice({
  children,
  className = "",
  polite = false
}: {
  children: React.ReactNode;
  className?: string;
  polite?: boolean;
}) {
  return <p className={`inline-notice ${className}`} role={polite ? "status" : "alert"}>{children}</p>;
}

export function installGlobalErrorLogging() {
  window.addEventListener("error", (event) => {
    logClientEvent("error", "client.window.error", {
      component: "window",
      error_name: event.error instanceof Error ? event.error.name : "Error",
      error_message: event.message,
      route: window.location.pathname
    });
  });
  window.addEventListener("unhandledrejection", (event) => {
    const reason = event.reason;
    logClientEvent("error", "client.window.unhandled_rejection", {
      component: "window",
      error_name: reason instanceof Error ? reason.name : "UnhandledRejection",
      error_message: reason instanceof Error ? reason.message : String(reason),
      route: window.location.pathname
    });
  });
}

export function mountApp() {
  installGlobalErrorLogging();
  ReactDOM.createRoot(document.getElementById("root")!).render(<App />);
}
