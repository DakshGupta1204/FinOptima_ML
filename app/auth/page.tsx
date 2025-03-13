"use client";
import React, { useState, useEffect } from "react";
import {
  Mail,
  Lock,
  Settings,
  Sun,
  Moon,
  BarChart as ChartBar,
  Wallet,
  PiggyBank,
  LineChart,
  Target,
  Brain,
  Shield,
  Eye,
  EyeOff,
  User,
} from "lucide-react";
import { loginUser, registerUser } from "@/services/authService";
import toast from "react-hot-toast";

const Auth = () => {
  const [darkMode, setDarkMode] = useState(true);
  const [isSignUp, setIsSignUp] = useState(false); // Toggle between Login & Signup
  const [type, setType] = useState("password");
  const [eye, setEye] = useState(<Eye />);
  const [loading,setLoading] = useState(false);
  const [error,setError] = useState("");
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add("dark");
    } else {
      document.documentElement.classList.remove("dark");
    }
  }, [darkMode]);

  const toggleDarkMode = () => {
    setDarkMode(!darkMode);
  };

  const showHide = () => {
    if (type === "password") {
      setType("text");
      setEye(<EyeOff />);
    } else {
      setType("password");
      setEye(<Eye />);
    }
  };
  const handleSubmit = async(e:any) =>{
    e.preventDefault();
    setLoading(true);
    setError("");

    try {
      if (isSignUp) {
        await registerUser(name, email, password);
        toast.success("Account created successfully, You can now login !");
        setIsSignUp(false);
      } else {
        const response = await loginUser(email, password);
        localStorage.setItem("token", response.token);
        toast.success("Logged in successfully");
        setTimeout(() => {
          window.location.href = "/dashboard";
        }
        , 1000);

      }
    } catch (err:any) {
      setError(err);
    } finally {
      setLoading(false);
    }
  }
  const backgroundIcons = [
    { Icon: Mail, size: 200, top: 20, left: 40, rotate: "rotate-12" },
    { Icon: Lock, size: 150, bottom: 40, right: 60, rotate: "-rotate-12" },
    { Icon: ChartBar, size: 180, top: "25%", left: "22%", rotate: "rotate-45" },
    { Icon: Wallet, size: 160, top: "10%", right: "5%", rotate: "-rotate-15" },
    { Icon: PiggyBank, size: 140, bottom: "5%", left: "20%", rotate: "rotate-30" },
    { Icon: LineChart, size: 170, top: "10%", right: "30%", rotate: "-rotate-20" },
    { Icon: Target, size: 130, bottom: "40%", right: "15%", rotate: "rotate-25" },
    { Icon: Brain, size: 150, top: "60%", left: "5%", rotate: "-rotate-35" },
    { Icon: Shield, size: 120, bottom: "5%", left: "55%", rotate: "rotate-15" },
  ];

  return (
    <div className="w-full min-h-screen flex items-center justify-center bg-gray-50 dark:bg-[#0d1117] relative overflow-hidden p-4 transition-colors duration-300">
      {/* Background Icon Decorations */}
      <div className="absolute inset-0 opacity-20 dark:opacity-10">
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

      {/* Auth Card */}
      <div className="w-full max-w-sm bg-white dark:bg-[#161b22] rounded-xl shadow-lg dark:shadow-2xl p-8 relative z-10 transition-all duration-300">
        {/* Header */}
        <div className="flex justify-between items-center mb-6">
          <h1 className="text-2xl font-semibold text-gray-800 dark:text-gray-200">
            {isSignUp ? "Sign Up" : "Login"}
          </h1>
          <button
            onClick={toggleDarkMode}
            className="p-2 rounded-lg bg-gray-100 dark:bg-gray-800 text-primary dark:text-primary-light hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
          >
            {darkMode ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
          </button>
        </div>

        {/* Form Fields */}
        <form className="space-y-5" onSubmit={handleSubmit}>
          {isSignUp && (
            <div className="relative group">
              <input
                type="text"
                placeholder="Full Name"
                value={name}
                onChange={(e) => setName(e.target.value)}
                required
                className="w-full px-4 py-3 bg-transparent border border-gray-300 dark:border-primary focus:border-primary dark:focus:border-primary-light rounded-lg focus:outline-none focus:ring-2 focus:ring-primary/20 dark:focus:ring-primary-light/20 text-gray-800 dark:text-gray-200 transition-all duration-300"
              />
              <User className="absolute right-3 top-3 text-gray-400 dark:text-primary-light transition-colors duration-300" />
            </div>
          )}

          <div className="relative group">
            <input
              type="email"
              placeholder="Email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              className="w-full px-4 py-3 bg-transparent border border-gray-300 dark:border-primary focus:border-primary dark:focus:border-primary-light rounded-lg focus:outline-none focus:ring-2 focus:ring-primary/20 dark:focus:ring-primary-light/20 text-gray-800 dark:text-gray-200 transition-all duration-300"
            />
            <Mail className="absolute right-3 top-3 text-gray-400 dark:text-primary-light transition-colors duration-300" />
          </div>

          <div className="relative group">
            <input
              type={type}
              placeholder="Password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              className="w-full px-4 py-3 bg-transparent border border-gray-300 dark:border-primary focus:border-primary dark:focus:border-primary-light rounded-lg focus:outline-none focus:ring-2 focus:ring-primary/20 dark:focus:ring-primary-light/20 text-gray-800 dark:text-gray-200 transition-all duration-300"
            />
            <button
              type="button"
              onClick={showHide}
              className="absolute right-3 top-3 text-gray-400 dark:text-primary-light transition-colors duration-300"
            >
              {eye}
            </button>
          </div>

          {isSignUp && (
            <div className="relative group">
              <input
                type={type}
                placeholder="Confirm Password"
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
                required
                className="w-full px-4 py-3 bg-transparent border border-gray-300 dark:border-primary focus:border-primary dark:focus:border-primary-light rounded-lg focus:outline-none focus:ring-2 focus:ring-primary/20 dark:focus:ring-primary-light/20 text-gray-800 dark:text-gray-200 transition-all duration-300"
              />
              <Lock className="absolute right-3 top-3 text-gray-400 dark:text-primary-light transition-colors duration-300" />
            </div>
          )}

          {error && <p className="text-red-500 text-sm">{error}</p>}

          {/* Submit Button */}
          <button
            type="submit"
            className="w-full py-3 bg-primary dark:bg-primary-light text-white dark:text-black font-medium rounded-lg hover:bg-primary-dark dark:hover:bg-primary-dark dark:hover:text-white transition-all duration-300 transform hover:scale-[1.02]"
            disabled={loading}
          >
            {loading ? "Processing..." : isSignUp ? "Sign Up" : "Sign In"}
          </button>

          {/* Toggle between Login & Signup */}
          <p className="text-center text-sm text-gray-600 dark:text-gray-400">
            {isSignUp ? "Already have an account?" : "Don't have an account?"}
            <button
              type="button"
              onClick={() => setIsSignUp(!isSignUp)}
              className="text-primary dark:text-primary-light hover:text-primary-dark dark:hover:text-white ml-1 transition-colors"
            >
              {isSignUp ? "Login" : "Sign Up"}
            </button>
          </p>
        </form>
      </div>
    </div>
  );
};

export default Auth;
