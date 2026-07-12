import { Navigate, Route, Routes } from "react-router-dom";
import { WorkspacePage } from "./pages/WorkspacePage";

export default function App() { return <Routes><Route path="/" element={<WorkspacePage />} /><Route path="/sessions/:sessionId" element={<WorkspacePage />} /><Route path="*" element={<Navigate to="/" replace />} /></Routes>; }
