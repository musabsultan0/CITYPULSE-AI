import type { Metadata } from "next";
import "./globals.css";
export const metadata: Metadata = { title: "CITYPULSE AI", description: "Smart city multi-agent operations prototype" };
export default function RootLayout({children}:{children:React.ReactNode}) { return <html lang="en"><body>{children}</body></html>; }
