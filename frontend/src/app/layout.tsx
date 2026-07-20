import type {
  Metadata,
  Viewport,
} from "next";

import "./globals.css";
import "./auth.css";
import "./history.css";

export const metadata: Metadata = {
  title:
    "Enterprise Banking Policy Copilot",

  description:
    "Permission-aware banking policy and " +
    "compliance assistant.",
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}