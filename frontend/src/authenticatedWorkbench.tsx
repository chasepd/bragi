import { QueryClientProvider } from "@tanstack/react-query";
import { useEffect } from "react";
import type { AuthenticatedWorkbenchProps } from "./appShell";
import { Workbench, workbenchQueryClient } from "./workbenchCore";

workbenchQueryClient.clear();

export default function AuthenticatedWorkbench(props: AuthenticatedWorkbenchProps) {
  useEffect(() => () => workbenchQueryClient.clear(), []);
  return (
    <QueryClientProvider client={workbenchQueryClient}>
      <Workbench {...props} />
    </QueryClientProvider>
  );
}
