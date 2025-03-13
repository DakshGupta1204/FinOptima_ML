"use client";
import React, { useState } from "react";
import axios from "axios";
import {
  User,
  LogOut,
  LineChart,
  Calculator,
  Bot,
  ShieldCheck,
  PiggyBank,
  Moon,
  Sun,
  Mail,
  ChartBar,
  Wallet,
  Target,
  Brain,
  Shield,
  Lock,
  ChevronDown,
} from "lucide-react";
import { Card } from "@/components/ui/card";
import { Bar } from "react-chartjs-2";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Button } from "@/components/ui/button";
import { useRouter } from "next/navigation";

const RiskAssessment: React.FC = () => {
  const router = useRouter();
  const [formData, setFormData] = useState({
    risk_tolerance: "",
    investment_horizon: "",
    market_experience: "",
    income_stability: "",
    emergency_fund: "",
  });

  const [results, setResults] = useState<any>(null);
  const [tickers, setTickers] = useState<string>("");
  const [riskAversion, setRiskAversion] = useState<number>(0.1);
  const [mlTool, setMlTool] = useState<string | null>("Mean-Variance Optimization");
  const [darkMode, setDarkMode] = useState(false); // Theme State
  const mlTools = [
    "Mean-Variance Optimization",
    "Deep Reinforcement Learning",
    "Black-Litterman Model",
    "Genetic Algorithm",
  ];

  const handleOptimize = async () => {
    const tickersArray = tickers.split(",").map((t) => t.trim().toUpperCase());
    try {
      const response = await axios.post("http://localhost:8000/portfolio-optimization", {
        tickers: tickersArray,
        risk_aversion: riskAversion,
      });
      setResults(response.data);
    } catch (err) {
      console.error("Error optimizing portfolio:", err);
    }
  };

  const toggleDarkMode = () => {
    setDarkMode(!darkMode);
    if (!darkMode) {
      document.documentElement.classList.add("dark");
    } else {
      document.documentElement.classList.remove("dark");
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const backgroundIcons = [
    { Icon: Mail, size: 200, top: 60, left: 40, rotate: "rotate-12" },
    { Icon: Lock, size: 150, bottom: 40, right: 60, rotate: "-rotate-12" },
    { Icon: ChartBar, size: 180, top: "7%", left: "40%", rotate: "rotate-135" },
    { Icon: Wallet, size: 160, top: "30%", right: "1%", rotate: "-rotate-15" },
    { Icon: PiggyBank, size: 140, bottom: "1%", left: "20%", rotate: "rotate-30" },
    { Icon: LineChart, size: 170, top: "2%", right: "18%", rotate: "-rotate-20" },
    { Icon: Target, size: 130, bottom: "30%", right: "25%", rotate: "rotate-25" },
    { Icon: Brain, size: 150, top: "60%", left: "1%", rotate: "-rotate-35" },
    { Icon: Shield, size: 120, bottom: "5%", left: "55%", rotate: "rotate-15" },
  ];

  const [dropdownOpen, setDropdownOpen] = useState(false);

  return (
    <div className="min-h-screen bg-gray-100 text-gray-900 dark:bg-[#0d1117] dark:text-white">
      <div className="absolute inset-0 opacity-20 dark:opacity-10 ">
        {backgroundIcons.map(({ Icon, size, ...style }, idx) => (
          <Icon
            key={idx}
            size={size}
            className={`text-primary-dark dark:text-primary-light absolute ${style.rotate} transition-all duration-500`}
            style={{
              top: style.top,
              left: style.left,
              bottom: style.bottom,
              right: style.right,
            }}
          />
        ))}
      </div>
      {/* Navbar */}
      <nav className="flex justify-between items-center p-16 z-20 relative">
        <h1 className="text-2xl font-bold flex items-center justify-center gap-2 text-primary dark:text-white cursor-pointer" onClick={() => router.push("/")}>
          <LineChart size={32} className="dark:text-primary-light" /> FinOptima
        </h1>

        {/* ML Tools in Navbar */}

        <div className="flex items-center space-x-4">
          {/* Theme Toggle */}
          <div className="relative">
            <button
              onClick={() => setDropdownOpen(!dropdownOpen)}
              className="flex items-center bg-gray-200 dark:bg-gray-800 px-3 py-2 rounded-lg hover:bg-gray-300 dark:hover:bg-gray-700 transition"
            >
              <Bot size={20} className="mr-2" />
              ML Tools
              <ChevronDown size={18} className="ml-2" />
            </button>

            {dropdownOpen && (
              <div className="absolute right-0 mt-2 w-56 bg-white dark:bg-gray-900 shadow-md rounded-md py-2 z-50">
                {[
                  { name: "Risk Assessment", icon: LineChart, link: "/ml-risk" },
                  { name: "Portfolio Optimiser", icon: Calculator, link:"/ml-portfolio" },
                  { name: "Market Analysis", icon: Bot, link:"/ml-market" },
                  { name: "Financial Planning", icon: ShieldCheck, link:"/ml-financial" },
                  { name: "Investment Planner", icon: PiggyBank, link:"/ml-investment" },
                ].map(({ name, icon: Icon, link }, idx) => (
                  <button
                    key={idx}
                    className="flex items-center px-4 py-2 w-full text-left hover:bg-gray-100 dark:hover:bg-gray-700 transition"
                    onClick={() => router.push(link)}
                  >
                    <Icon size={18} className="mr-3" />
                    {name}
                  </button>
                ))}
              </div>
            )}
          </div>
          <button
            onClick={toggleDarkMode}
            className="p-2 rounded-md hover:bg-gray-300 transition bg-gray-200 dark:bg-gray-800 dark:hover:bg-gray-600 text-primary dark:text-primary-light"
          >
            {darkMode ? <Sun size={22} /> : <Moon size={22} />}
          </button>

          {/* Profile & Logout */}
          <button className="flex items-center space-x-1 bg-primary px-3 py-2 rounded-lg hover:bg-primary-dark transition text-white" onClick={() => router.push("/user-profile")}>
            <User size={20} />
            <span>Profile</span>
          </button>
          <button className="flex items-center space-x-1 bg-gray-200 px-3 py-2 rounded-lg hover:bg-red-500 transition hover:text-white dark:bg-gray-900 dark:hover:bg-red-600">
            <LogOut size={20} />
            <span>Logout</span>
          </button>
        </div>
      </nav>
      {/* Main Content */}
      <div className="max-w-4xl mx-auto p-8 relative z-10">
        {/* Input Section */}
        <Card className="p-6 mb-6">
          <h1 className="text-2xl font-bold mb-4">ML Portfolio Optimizer</h1>
          <label className="block text-sm font-medium">Stock Tickers (comma-separated)</label>
          <Input
            value={tickers}
            onChange={(e) => setTickers(e.target.value)}
            placeholder="e.g., AAPL, MSFT, GOOG"
            className="my-5"
          />

          <label className="block text-sm font-medium">Risk Aversion (0 - 1)</label>
          <Input
            type="number"
            step="0.01"
            min="0"
            max="1"
            value={riskAversion}
            onChange={(e) => setRiskAversion(parseFloat(e.target.value))}
            className="my-5"
          />

          <label className="block text-sm font-medium">ML Optimization Model</label>
          <Select onValueChange={(value) => setMlTool(value)}>
            <SelectTrigger className="my-5">
              <SelectValue placeholder="Select an ML Tool" />
            </SelectTrigger>
            <SelectContent>
              {mlTools.map((tool) => (
                <SelectItem key={tool} value={tool}>
                  {tool}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>

          <Button onClick={handleOptimize} className="w-full text-white">Run Optimization</Button>
        </Card>

        {/* Results Section */}
        {results && (
          <Card className="p-4">
            <h2 className="text-xl font-semibold mb-3">Optimization Results</h2>

            <div className="mb-3">
              <p><strong>Sharpe Ratio:</strong> {results.risk_metrics.sharpe_ratio.toFixed(2)}</p>
              <p><strong>Annual Return:</strong> {results.risk_metrics.annual_return.toFixed(2)}</p>
              <p><strong>Max Drawdown:</strong> {results.risk_metrics.max_drawdown.toFixed(2)}</p>
            </div>

            {/* Portfolio Weights Chart */}
            <div className="mt-4">
              <h3 className="text-lg font-medium mb-2">Portfolio Allocation</h3>
              <Bar
                data={{
                  labels: Object.keys(results.weights),
                  datasets: [
                    {
                      label: "Portfolio Weights",
                      data: Object.values(results.weights),
                      backgroundColor: "rgba(75, 192, 192, 0.6)",
                    },
                  ],
                }}
              />
            </div>
          </Card>
        )}
      </div>
    </div>
  );
};

export default RiskAssessment;
