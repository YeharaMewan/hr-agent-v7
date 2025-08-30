# Rise HR Agent - React Frontend

A modern React application built with Vite that provides an intelligent HR chat interface. This frontend integrates with the FastAPI backend to deliver AI-powered HR assistance.

## 🚀 Features

- **Modern React UI** - Built with React 18 and Vite for fast development
- **Dark Theme** - Sleek dark interface with smooth animations
- **Real-time Chat** - Streaming responses for immediate feedback
- **Chart Integration** - Data visualization using Chart.js
- **Responsive Design** - Works on desktop and mobile devices
- **No Authentication** - Simplified implementation focused on core functionality

## 📦 Project Structure

```
frontend/
├── src/
│   ├── components/
│   │   ├── LandingView/       # Landing page with logo and input
│   │   ├── ChatView/          # Main chat interface
│   │   ├── MessageBubble/     # Individual message components
│   │   ├── SuggestionChips/   # Interactive suggestion buttons
│   │   └── ChartRenderer/     # Chart visualization component
│   ├── hooks/
│   │   ├── useApi.js          # API integration hook
│   │   └── useStreaming.js    # Streaming response hook
│   ├── services/
│   │   └── api.js             # Backend API service layer
│   ├── styles/
│   │   └── globals.css        # Global styles and theme
│   └── App.jsx                # Main application component
├── public/
│   └── RiseHRLogo.png         # Company logo
└── package.json
```

## 🛠️ Installation

1. **Navigate to the frontend directory**
   ```bash
   cd Agent/frontend
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Start the development server**
   ```bash
   npm run dev
   ```

4. **Open your browser**
   ```
   http://localhost:5173
   ```

## 🔧 Backend Integration

This frontend is designed to work with the FastAPI backend in the `Agent/backend` directory.

### Backend Setup

1. **Navigate to backend directory**
   ```bash
   cd ../backend
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   # Create a .env file with:
   OPENAI_API_KEY=your_openai_api_key_here
   ```

4. **Run the backend server**
   ```bash
   python main.py
   ```

The backend will run on `http://localhost:8000`

### API Endpoints

- `POST /invoke` - Send regular queries
- `POST /invoke-stream` - Send streaming queries
- `GET /health` - Health check
- `GET /capabilities` - System capabilities

## 💻 Usage

1. **Landing Page**: Start by typing a question or clicking on suggestion chips
2. **Chat Interface**: Continue the conversation with follow-up questions
3. **Suggestions**: Use the lightbulb button to toggle suggestion chips
4. **Charts**: Some responses will include interactive data visualizations

## 🎨 Customization

### Theme Colors
Edit `src/styles/globals.css` to customize the dark theme:

```css
:root {
    --bg-color: #121212;
    --accent-color: #a78bfa;
    --text-color: #e4e4e7;
    /* ... other variables */
}
```

### API Configuration
Update the API base URL in `src/services/api.js`:

```javascript
const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000';
```

## 🧪 Development

### Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build
- `npm run lint` - Run ESLint (if configured)

### Adding New Components

1. Create a new directory in `src/components/`
2. Add your component file (`.jsx`)
3. Export from the component directory
4. Import in parent components

## 🔍 Troubleshooting

### Frontend Issues
- **Blank page**: Check browser console for errors
- **Components not loading**: Verify all imports are correct
- **Styles not applying**: Ensure `globals.css` is imported in `App.jsx`

### Backend Integration Issues
- **API errors**: Check if backend is running on port 8000
- **CORS issues**: Backend includes CORS middleware
- **Streaming not working**: Verify `/invoke-stream` endpoint is accessible

### Common Solutions
1. **Clear browser cache** if styles aren't updating
2. **Restart development server** after config changes
3. **Check console logs** for detailed error messages

## 🚀 Production Build

1. **Build the application**
   ```bash
   npm run build
   ```

2. **Serve the built files**
   ```bash
   npm run preview
   ```

The built files will be in the `dist/` directory.

## 📧 Support

If you encounter issues:
1. Check the console for error messages
2. Verify backend is running and accessible
3. Ensure all dependencies are installed
4. Check network connectivity for API calls

---

**Built with React + Vite for optimal performance and development experience**
