import { Navigate, Route, Routes } from "react-router-dom";
import { Tooltip } from "radix-ui";
import { WorkspacePage } from "./pages/WorkspacePage";

export default function App() { return <Tooltip.Provider><Routes><Route path="/" element={<WorkspacePage />} /><Route path="/sessions/:sessionId" element={<WorkspacePage />} /><Route path="*" element={<Navigate to="/" replace />} /></Routes></Tooltip.Provider>; }
