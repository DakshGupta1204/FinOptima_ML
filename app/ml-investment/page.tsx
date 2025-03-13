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
  const [userData, setUserData] = useState({
    age: 30,
    income: 50000,
    risk_tolerance: 0.5,
    investment_horizon: 5,
    financial_knowledge: 3,
    investment_goals: ["wealth_growth"],
  });

  const [simulationData, setSimulationData] = useState<any>(null);

  const [learningData, setLearningData] = useState<any>(null);

  const [mathData, setMathData] = useState<any>(null);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    setUserData({ ...userData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async () => {
    try {
      const response = await axios.post("http://127.0.0.1:8000/investment-learning", userData);
      setSimulationData(response.data.simulation_results.results);
      setMathData(response.data.simulation_results.metrics);
      setLearningData(response.data.learning_path);
      console.log(response.data);
    } catch (error) {
      console.error("Error fetching data:", error);
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
            <Logo/> FinOptima
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
      <div className="container mx-auto p-4 relative z-10 bg-white dark:bg-[#1a1a1a] rounded-lg shadow-lg p-6">
            <h1 className="text-2xl font-bold mb-4">Investment Learning System</h1>
            
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label>Age:</label>
                <input type="number" name="age" value={userData.age} onChange={handleChange} className="border p-2 w-full dark:bg-gray-800"/>
              </div>
      
              <div>
                <label>Income:</label>
                <input type="number" name="income" value={userData.income} onChange={handleChange} className="border p-2 w-full dark:bg-gray-800"/>
              </div>
      
              <div>
                <label>Risk Tolerance:</label>
                <input type="number" step="0.1" name="risk_tolerance" value={userData.risk_tolerance} onChange={handleChange} className="border p-2 w-full dark:bg-gray-800"/>
              </div>
      
              <div>
                <label>Investment Horizon (years):</label>
                <input type="number" name="investment_horizon" value={userData.investment_horizon} onChange={handleChange} className="border p-2 w-full dark:bg-gray-800"/>
              </div>
      
              <div>
                <label>Financial Knowledge (1-5):</label>
                <input type="number" name="financial_knowledge" value={userData.financial_knowledge} onChange={handleChange} className="border p-2 w-full dark:bg-gray-800"/>
              </div>
      
              <button onClick={handleSubmit} className="bg-primary hover:bg-primary-dark text-white px-4 py-2 rounded">Run Simulation</button>
            </div>
      
            {simulationData && (
              <div className="mt-8">
                <h2 className="text-xl font-bold">Simulation Results</h2>
                <LineChart width={1200} height={300} data={simulationData.map((value: number, index: number) => ({ index, value }))}>
                  <CartesianGrid strokeDasharray="30 3" />
                  <XAxis dataKey="index" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Line type="monotone" dataKey="value" stroke="#8884d8" />
                </LineChart>
              </div>
            )}
          </div>
    </div>
  );
};

export default RiskAssessment;
