"use client";
import React, { useState } from "react";
import axios from "axios";
import {
  User,
  LogOut,
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
import { LineChart as Logo } from "lucide-react";
import { CartesianGrid, Line, XAxis, YAxis, Tooltip, Legend, LineChart } from "recharts";
import { useRouter } from "next/navigation";

const RiskAssessment: React.FC = () => {
  const router = useRouter();
  const [darkMode, setDarkMode] = useState(false); // Theme State
  const [formData, setFormData] = useState({
    risk_tolerance: "",
    investment_horizon: "",
    market_experience: "",
    income_stability: "",
    emergency_fund: "",
  });

  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError("");

    try {
      const response = await axios.post("http://localhost:8000/risk-assessment", formData);
      setResult(response.data);
    } catch (err) {
      setError("Error fetching risk assessment. Please try again.");
    } finally {
      setLoading(false);
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
    <div className="min-h-screen bg-gray-100 text-gray-900 dark:bg-[#0d1117] dark:text-white z-0">
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
          <Logo /> FinOptima
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
                  { name: "Risk Assessment", icon: Logo, link: "/ml-risk" },
                  { name: "Portfolio Optimiser", icon: Calculator, link: "/ml-portfolio" },
                  { name: "Market Analysis", icon: Bot, link: "/ml-market" },
                  { name: "Financial Planning", icon: ShieldCheck, link: "/ml-financial" },
                  { name: "Investment Planner", icon: PiggyBank, link: "/ml-investment" },
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
      <div className="flex justify-around mx-16 px-4 gap-2 py-2">

      <div className="max-w-4xl mx-auto p-6 bg-white shadow-lg rounded-lg relative z-10 dark:bg-[#161b22] dark:text-white w-[80%]">
        <h2 className="text-2xl font-bold mb-4">Investment Risk Assessment</h2>
        <form onSubmit={handleSubmit} className="space-y-4">
          {Object.keys(formData).map((key) => (
            <div key={key}>
              <label className="block font-semibold capitalize">
                {key.replace("_", " ")}
              </label>
              <input
                type="number"
                name={key}
                value={formData[key as keyof typeof formData]}
                onChange={handleChange}
                className="w-full p-2 border border-green-400 rounded mt-1 focus:outline-none focus:ring-2 focus:ring-green-500 dark:border-gray-600 dark:bg-gray-800 dark:text-white"
                min={1}
                max={5}
                required
              />
              <p className="text-sm text-gray-500 mt-1">
                {key === "risk_tolerance" && "1 = Very Low, 5 = Very High"}
                {key === "investment_horizon" && "1 = Short-term (Less than a year), 5 = Long-term (10+ years)"}
                {key === "market_experience" && "1 = No experience, 5 = Highly experienced"}
                {key === "income_stability" && "1 = Unstable, 5 = Highly Stable"}
                {key === "emergency_fund" && "1 = No savings, 5 = Over 6 months of expenses saved"}
              </p>
            </div>
          ))}

          <button
            type="submit"
            className="w-full bg-green-600 text-white py-2 rounded-md hover:bg-green-800 transition"
            disabled={loading}
          >
            {loading ? "Assessing..." : "Get Risk Assessment"}
          </button>
        </form>

        {error && <p className="text-red-500 mt-3">{error}</p>}

        

      </div>
      {result && (
          <div className="mt-5 p-6 bg-white rounded-md dark:bg-[#161b22] dark:text-white h-[fit-content] mt-auto mb-auto z-10">
            <h3 className="text-2xl font-bold mb-8">Assessment Results</h3>
            <p>
              <strong>Risk Category:</strong> {result.risk_category}
              <span className="block text-sm text-gray-500 mt-2 mb-4">
                {result.risk_category === "Low" && "You prefer safe investments with minimal risk."}
                {result.risk_category === "Moderate" && "You can tolerate some risk for better returns."}
                {result.risk_category === "High" && "You are open to high-risk investments for potential high rewards."}
              </span>
            </p>

            <p>
              <strong>Confidence Score:</strong> {result.confidence_score.toFixed(2)}%
              <span className="block text-sm text-gray-500 mt-2 mb-4">
                This shows how certain the AI model is about your risk category.
              </span>
            </p>

            <p>
              <strong>Risk Score:</strong> {result.risk_score.toFixed(2)}
              <span className="block text-sm text-gray-500 mt-2 mb-4">
                A lower score means lower risk-taking tendency, while a higher score means higher risk-taking behavior.
              </span>
            </p>

            <p>
              <strong>Market Adjusted Score:</strong> {result.market_adjusted_score.toFixed(2)}
              <span className="block text-sm text-gray-500 mt-2 mb-4">
                This adjusts your risk score based on current market trends.
              </span>
            </p>

            <p className="text-sm text-gray-500">
              Generated on: {new Date(result.assessment_timestamp).toLocaleString()}
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default RiskAssessment;
