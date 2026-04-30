package com.crispasr.demo

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.viewModels
import com.crispasr.demo.ui.main.MainScreen
import com.crispasr.demo.ui.main.MainScreenViewModel
import com.crispasr.demo.ui.theme.CrispASRDemoTheme

class MainActivity : ComponentActivity() {
    private val viewModel: MainScreenViewModel by viewModels { MainScreenViewModel.factory() }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            CrispASRDemoTheme {
                MainScreen(viewModel)
            }
        }
    }
}
